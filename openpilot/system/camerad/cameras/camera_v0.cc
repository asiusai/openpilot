#include "system/camerad/cameras/camera_common.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/media.h>
#include <linux/media-bus-format.h>
#include <linux/v4l2-controls.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "system/camerad/cameras/hw.h"
#include "system/camerad/cameras/autoexposure.h"
#include "system/camerad/cameras/ife.h"
#include "system/camerad/cameras/cdm_parse.h"
#include "system/camerad/cameras/nv12_info.h"
#include "system/camerad/sensors/sensor.h"


ExitHandler do_exit;

struct OneCamRoute {
  int csiphy;
  int csid;
  int vfe;
  const char *sensor;
};

// Asius v0 camera routing:
//   Runtime camerad path: CSIPHY -> CSID PIX pad -> VFE PIX -> NV12 DMABUF.
//   Raw RDI probing lives in standalone bring-up tools, not in this runtime path.
// openpilot camera_num 0 is wide road, camera_num 1 is road,
// camera_num 2 is driver. For Asius v0:
//   CAM1 -> driver, CAM2 -> road, CAM3 -> wide road.
struct V0CamConfig {
  uint32_t csiphy_entity;
  uint32_t csid_entity;
  uint32_t vfe_pix_entity;
  int pix_video_dev;
  const char *sensor_name;
  int csiphy_subdev;
  int csid_subdev;
  int vfe_pix_subdev;
};

static int find_v4l_dev(const char *prefix, const char *name) {
  for (int i = 0; i < 64; i++) {
    auto path = util::string_format("/sys/class/video4linux/%s%d/name", prefix, i);
    auto dev_name = util::read_file(path);
    if (!dev_name.empty() && dev_name.find(name) == 0) return i;
  }
  return -1;
}

static uint32_t find_media_entity(int media_fd, const char *name) {
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;
    if (strcmp(ent.name, name) == 0) return ent.id;
  }
  return 0;
}

static V0CamConfig resolve_cam_config(int media_fd, int cam_idx) {
  static const OneCamRoute routing[] = {
    {3, 1, 1, "os04c10 20-0036"},
    {2, 0, 0, "os04c10 18-0036"},
    {0, 2, 2, "os04c10 16-0036"},
  };
  const auto *r = &routing[cam_idx];
  V0CamConfig cfg = {};
  cfg.pix_video_dev = -1;
  cfg.csiphy_subdev = -1;
  cfg.csid_subdev = -1;
  cfg.vfe_pix_subdev = -1;
  cfg.sensor_name = r->sensor;
  cfg.csiphy_entity = find_media_entity(media_fd, util::string_format("msm_csiphy%d", r->csiphy).c_str());
  cfg.csid_entity = find_media_entity(media_fd, util::string_format("msm_csid%d", r->csid).c_str());
  cfg.vfe_pix_entity = find_media_entity(media_fd, util::string_format("msm_vfe%d_pix", r->vfe).c_str());
  cfg.pix_video_dev = find_v4l_dev("video", util::string_format("msm_vfe%d_video3", r->vfe).c_str());
  cfg.vfe_pix_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_vfe%d_pix", r->vfe).c_str());

  cfg.csiphy_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csiphy%d", r->csiphy).c_str());
  cfg.csid_subdev = find_v4l_dev("v4l-subdev", util::string_format("msm_csid%d", r->csid).c_str());
  return cfg;
}

static V0CamConfig v0_cams[3];


struct OneSensorRegWrite {
  uint16_t addr;
  uint16_t data;
};

struct OneSensorWriteRegsCmd {
  uint64_t regs;
  uint32_t count;
  uint8_t data_width;
  uint8_t pad[3];
};

#define ONE_SENSOR_WRITE_REGS _IOW('S', 1, struct OneSensorWriteRegsCmd)

using VfeRegWrite = IspRegWrite;

struct VfeWriteRegsCmd {
  uint64_t regs;
  uint32_t count;
  uint32_t pad;
};

struct VfeDmiCmd {
  uint32_t dmi_cfg_offset;
  uint8_t ram_select;
  uint8_t pad[3];
  uint32_t count;
  uint64_t data;
};

#define VFE_IOC_MAGIC '#'
#define VFE_WRITE_REGS _IOW(VFE_IOC_MAGIC, 1, struct VfeWriteRegsCmd)
#define VFE_WRITE_DMI _IOW(VFE_IOC_MAGIC, 2, struct VfeDmiCmd)
#define VFE_REG_UPDATE _IO(VFE_IOC_MAGIC, 6)

static int v0_ioctl(int fd, unsigned long request, void *arg = nullptr) {
  int ret;
  int try_cnt = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR && try_cnt++ < 100);
  return ret;
}

// The mainline CAMSS graph links VFE_LINE_PIX from CSID source pad 4.
// The CSID-gen2 driver still programs that PIX/IPP path for sensor VC0.
static constexpr int ONE_PIX_CSID_SOURCE_PAD = 4;

static constexpr uint32_t ONE_SENSOR_DELAY_MS = 0xffffffffU;
static constexpr uint32_t OS04_RAW10_20FPS_VTS = 0x1275;

static const std::vector<i2c_random_wr_payload> &os04_default_init_regs() {
  static const std::vector<i2c_random_wr_payload> regs = [] {
    std::vector<i2c_random_wr_payload> result(std::begin(init_array_os04c10), std::end(init_array_os04c10));
    // Two-lane RAW10 sensor mode. Retain this module's white balance in the
    // VFE, after black-level subtraction.
    result.insert(result.end(), {
      {0x0301, 0x84}, {0x0305, 0x5b}, {0x0306, 0x00}, {0x3016, 0x32},
      {0x3106, 0x25}, {0x3501, 0x04}, {0x3502, 0x40}, {0x3511, 0x01},
      {0x3512, 0x20}, {0x3660, 0x00}, {0x366a, 0x64}, {0x3698, 0x40},
      {0x369a, 0x18}, {0x369c, 0x14}, {0x36a1, 0x5d}, {0x370a, 0x00},
      {0x370e, 0x0c}, {0x3713, 0x00}, {0x3748, 0x00}, {0x374a, 0x00},
      {0x374c, 0x00}, {0x374e, 0x00}, {0x3757, 0x0e}, {0x377b, 0x20},
      {0x37be, 0x08}, {0x37c7, 0x08}, {0x3c8c, 0x20}, {0x4090, 0x14},
      {0x40ba, 0x00}, {0x4507, 0x64}, {0x4803, 0x10}, {0x480e, 0x00},
      {0x4813, 0x00}, {0x4823, 0x3c}, {0x4825, 0x32}, {0x484b, 0x07},
      {0x5036, 0x00}, {0x301c, 0xf0}, {0x301f, 0xd0}, {0x3022, 0x01},
      {0x3621, 0x90}, {0x3681, 0xa6}, {0x3682, 0x53}, {0x3683, 0x2a},
      {0x3684, 0x15}, {0x3706, 0x4a}, {0x370b, 0xa2}, {0x370f, 0x04},
      {0x3716, 0x24}, {0x3741, 0x4a}, {0x3743, 0x4a}, {0x3745, 0x4a},
      {0x3747, 0x4a}, {0x3749, 0xa2}, {0x374b, 0xa2}, {0x374d, 0xa2},
      {0x374f, 0xa2}, {0x378d, 0x30}, {0x3790, 0x4a}, {0x3791, 0xa2},
      {0x3798, 0xc0}, {0x37a1, 0x01}, {0x37a8, 0x01}, {0x380c, 0x04},
      {0x380d, 0x2e}, {0x380e, 0x12}, {0x380f, 0x75}, {0x3811, 0x09},
      {0x3813, 0x09}, {0x3820, 0x88}, {0x3880, 0x25}, {0x4809, 0x1e},
      {0x4837, 0x0a}, {0x4c01, 0x00}, {0x3822, 0x14}, {0x0100, 0x00},
      {0x5100, 0x04}, {0x5101, 0x00}, {0x5102, 0x04}, {0x5103, 0x00},
      {0x5104, 0x04}, {0x5105, 0x00}, {0x5140, 0x04}, {0x5141, 0x00},
      {0x5142, 0x04}, {0x5143, 0x00}, {0x5144, 0x04}, {0x5145, 0x00},
    });
    return result;
  }();
  return regs;
}

static void disable_csid_tpg(int csid_subdev, int cam_idx) {
  if (csid_subdev < 0) return;

  int fd = open(util::string_format("/dev/v4l-subdev%d", csid_subdev).c_str(), O_RDWR);
  if (fd < 0) return;

  struct v4l2_control ctrl = {};
  ctrl.id = V4L2_CID_TEST_PATTERN;
  ctrl.value = 0;
  if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) != 0) {
    LOGE("cam %d: disabling CSID TPG failed: %d (%s)", cam_idx, errno, strerror(errno));
  } else {
    LOG("cam %d: disabled CSID TPG", cam_idx);
  }

  close(fd);
}

static bool write_sensor_regs(int sensor_fd, const std::vector<i2c_random_wr_payload> &reg_array,
                              const char *name, int cam_idx) {
  if (reg_array.empty()) return true;

  std::vector<OneSensorRegWrite> regs;
  regs.reserve(reg_array.size());
  size_t regs_written = 0;

  auto flush_regs = [&]() {
    if (regs.empty()) return true;

    OneSensorWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)regs.data();
    cmd.count = regs.size();
    cmd.data_width = 1;

    int ret = HANDLE_EINTR(ioctl(sensor_fd, ONE_SENSOR_WRITE_REGS, &cmd));
    if (ret != 0) {
      LOGE("cam %d: failed to write %s sensor regs (%zu): %d (%s)",
           cam_idx, name, regs.size(), errno, strerror(errno));
      return false;
    }

    regs_written += regs.size();
    regs.clear();
    return true;
  };

  for (const auto &r : reg_array) {
    if (r.reg_addr == ONE_SENSOR_DELAY_MS) {
      if (!flush_regs()) return false;

      const uint32_t delay_ms = std::min(r.reg_data, 10000U);
      LOG("cam %d: delaying %u ms during %s sensor regs", cam_idx, delay_ms, name);
      usleep(delay_ms * 1000U);
      continue;
    }

    regs.push_back({(uint16_t)r.reg_addr, (uint16_t)r.reg_data});
    if (r.reg_addr == 0x0103) {
      if (!flush_regs()) return false;
      usleep(5000);
    }
  }

  if (!flush_regs()) return false;

  if (strcmp(name, "exposure") != 0) {
    LOG("cam %d: wrote %zu %s sensor regs", cam_idx, regs_written, name);
  }
  return true;
}

static bool apply_os04_20fps_timing(int sensor_fd, int cam_idx) {
  return write_sensor_regs(sensor_fd, {
    {0x380e, (OS04_RAW10_20FPS_VTS >> 8) & 0xff},
    {0x380f, OS04_RAW10_20FPS_VTS & 0xff},
  }, "20fps timing", cam_idx);
}

class OneCamera {
public:
  CameraConfig cc;
  std::unique_ptr<SensorInfo> sensor;
  bool enabled;

  int video_fd = -1;
  int sensor_fd = -1;
  bool streaming = false;

  int n_bufs = 4;
  std::unique_ptr<VisionBuf[]> vfe_buffers;

  uint32_t output_width = 0, output_height = 0;
  uint32_t stride = 0, y_height = 0, uv_height = 0, yuv_size = 0, uv_offset = 0;
  uint32_t vipc_stride = 0, vipc_y_height = 0, vipc_uv_height = 0, vipc_yuv_size = 0, vipc_uv_offset = 0;

  OneCamera(const CameraConfig &config) : cc(config), enabled(config.enabled) {}

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void setup_media_links();
  void set_formats();
  bool write_vfe_regs(const std::vector<VfeRegWrite> &regs, const char *name);
  bool apply_vfe_tuning();
  void queue_all_buffers();
  void stream_on();
  void stop_streaming();
  int dequeue_frame(uint64_t *timestamp);
  void queue_frame(int index);
  void set_exposure(int exposure_time, int gain_idx, bool dc_gain_enabled);

  VisionIpcServer *vipc_server = nullptr;
  VisionStreamType stream_type;
};


static void reset_all_media_links() {
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) return;

  for (int i = 0; i < 20; i++) {
    std::string vpath = util::string_format("/dev/video%d", i);
    int vfd = open(vpath.c_str(), O_RDWR);
    if (vfd >= 0) {
      int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      ioctl(vfd, VIDIOC_STREAMOFF, &type);
      type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(vfd, VIDIOC_STREAMOFF, &type);
      close(vfd);
    }
  }

  std::vector<uint32_t> csid_ents, vfe_ents, csiphy_ents;
  struct media_entity_desc ent = {};
  for (ent.id = 0 | MEDIA_ENT_ID_FLAG_NEXT; ; ent.id |= MEDIA_ENT_ID_FLAG_NEXT) {
    if (ioctl(media_fd, MEDIA_IOC_ENUM_ENTITIES, &ent) < 0) break;
    if (strncmp(ent.name, "msm_csiphy", 10) == 0) csiphy_ents.push_back(ent.id);
    if (strncmp(ent.name, "msm_csid", 8) == 0) csid_ents.push_back(ent.id);
    if (strncmp(ent.name, "msm_vfe", 7) == 0) vfe_ents.push_back(ent.id);
  }
  for (uint32_t csiphy : csiphy_ents) {
    for (uint32_t csid : csid_ents) {
      struct media_link_desc link = {};
      link.source = {.entity = csiphy, .index = 1};
      link.sink = {.entity = csid, .index = 0};
      link.flags = 0;
      ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link);
    }
  }
  for (uint32_t csid : csid_ents) {
    for (uint32_t vfe : vfe_ents) {
      for (int pad = 1; pad <= 4; pad++) {
        struct media_link_desc link = {};
        link.source = {.entity = csid, .index = (uint16_t)pad};
        link.sink = {.entity = vfe, .index = 0};
        link.flags = 0;
        ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link);
      }
    }
  }
  close(media_fd);
  LOG("reset all media links");
}

void OneCamera::setup_media_links() {
  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }

  int cam_idx = cc.camera_num;
  auto &dcfg = v0_cams[cam_idx];

  struct media_link_desc link = {};

  disable_csid_tpg(dcfg.csid_subdev, cam_idx);

  // CSIPHY -> CSID (source pad 1 -> sink pad 0)
  link.source = {.entity = (uint32_t)dcfg.csiphy_entity, .index = 1};
  link.sink = {.entity = (uint32_t)dcfg.csid_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csiphy->csid link FAILED: %d (%s)", cam_idx, errno, strerror(errno));
  memset(&link, 0, sizeof(link));

  // Mainline CAMSS exposes the PIX path on the CSID source pad selected here.
  link.source = {.entity = (uint32_t)dcfg.csid_entity, .index = ONE_PIX_CSID_SOURCE_PAD};
  link.sink = {.entity = (uint32_t)dcfg.vfe_pix_entity, .index = 0};
  link.flags = MEDIA_LNK_FL_ENABLED;
  if (ioctl(media_fd, MEDIA_IOC_SETUP_LINK, &link) != 0)
    LOGE("cam %d: csid->vfe PIX link FAILED: %d (%s)", cam_idx, errno, strerror(errno));

  close(media_fd);
  LOG("cam %d: media links set up (VFE PIX mode)", cam_idx);
}

void OneCamera::set_formats() {
  int cam_idx = cc.camera_num;
  auto &dcfg = v0_cams[cam_idx];
  constexpr uint32_t media_bus_code = MEDIA_BUS_FMT_SBGGR10_1X10;
  // set format on CSIPHY subdev
  int csiphy_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csiphy_subdev).c_str(), O_RDWR);
  if (csiphy_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csiphy_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    sfmt.pad = 1;
    ioctl(csiphy_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csiphy_fd);
  }

  // set format on CSID subdev
  int csid_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csid_subdev).c_str(), O_RDWR);
  if (csid_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);

    // PIX source pad matching the selected virtual channel.
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = ONE_PIX_CSID_SOURCE_PAD;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(csid_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
    close(csid_fd);
  }

  // set format on sensor subdev
  if (sensor_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(sensor_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);
  }

  int vfe_pix_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.vfe_pix_subdev).c_str(), O_RDWR);
  if (vfe_pix_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    ioctl(vfe_pix_fd, VIDIOC_SUBDEV_S_FMT, &sfmt);

    struct v4l2_subdev_selection sel = {};
    sel.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sel.pad = 0;
    sel.target = V4L2_SEL_TGT_COMPOSE;
    sel.r.left = 0;
    sel.r.top = 0;
    sel.r.width = output_width;
    sel.r.height = output_height;
    if (ioctl(vfe_pix_fd, VIDIOC_SUBDEV_S_SELECTION, &sel) != 0) {
      LOGE("cam %d: VFE PIX S_SELECTION compose failed: %d (%s)", cam_idx, errno, strerror(errno));
    }

    sfmt.pad = 1;
    sfmt.format.width = output_width;
    sfmt.format.height = output_height;
    sfmt.format.code = MEDIA_BUS_FMT_YUYV8_1_5X8;
    if (ioctl(vfe_pix_fd, VIDIOC_SUBDEV_S_FMT, &sfmt) != 0) {
      LOGE("cam %d: VFE PIX source S_FMT failed: %d (%s)", cam_idx, errno, strerror(errno));
    } else {
      LOG("cam %d: VFE PIX source format %ux%u code=0x%x",
          cam_idx, sfmt.format.width, sfmt.format.height, sfmt.format.code);
    }
    close(vfe_pix_fd);
  }

  struct v4l2_format vfmt = {};
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vfmt.fmt.pix_mp.width = output_width;
  vfmt.fmt.pix_mp.height = output_height;
  vfmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  vfmt.fmt.pix_mp.num_planes = 1;
  vfmt.fmt.pix_mp.plane_fmt[0].bytesperline = stride;
  vfmt.fmt.pix_mp.plane_fmt[0].sizeimage = yuv_size;
  if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) == 0) {
    stride = vfmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    y_height = output_height;
    uv_height = (output_height + 1) / 2;
    uv_offset = stride * y_height;
    yuv_size = vfmt.fmt.pix_mp.plane_fmt[0].sizeimage;
    LOG("cam %d: VFE PIX format set raw=%dx%d out=%dx%d stride=%u size=%u video_stride=%u sizeimage=%u",
        cam_idx, sensor->frame_width, sensor->frame_height, vfmt.fmt.pix_mp.width, vfmt.fmt.pix_mp.height,
        stride, yuv_size, vfmt.fmt.pix_mp.plane_fmt[0].bytesperline, vfmt.fmt.pix_mp.plane_fmt[0].sizeimage);
  } else {
    LOGE("cam %d: VFE PIX S_FMT failed: %d (%s)", cam_idx, errno, strerror(errno));
  }
}

bool OneCamera::write_vfe_regs(const std::vector<VfeRegWrite> &regs, const char *name) {
  if (video_fd < 0) return false;
  if (regs.empty()) return true;

  for (size_t offset = 0; offset < regs.size(); ) {
    const size_t count = std::min<size_t>(regs.size() - offset, 1024);
    VfeWriteRegsCmd cmd = {};
    cmd.regs = (uint64_t)(uintptr_t)(regs.data() + offset);
    cmd.count = count;
    if (v0_ioctl(video_fd, VFE_WRITE_REGS, &cmd) != 0) {
      LOGE("cam %d: failed to write %s VFE regs offset=%zu count=%zu: %d (%s)",
           cc.camera_num, name, offset, count, errno, strerror(errno));
      return false;
    }
    offset += count;
  }

  if (v0_ioctl(video_fd, VFE_REG_UPDATE) != 0) {
    LOGE("cam %d: failed to commit %s VFE regs: %d (%s)",
         cc.camera_num, name, errno, strerror(errno));
    return false;
  }

  LOG("cam %d: wrote %zu %s VFE regs", cc.camera_num, regs.size(), name);
  return true;
}

bool OneCamera::apply_vfe_tuning() {
  uint8_t program[8192] = {};
  std::vector<uint32_t> patches;
  const int size = build_initial_config(program, cc, sensor.get(), patches, output_width, output_height);
  std::vector<IspRegWrite> regs;
  std::vector<IspLutWrite> luts;
  if (!parse_isp_cdm(program, size, regs, luts)) return false;

  // Mainline CAMSS starts the ISP before userspace can configure it. Restore
  // the Bayer phase and full-frame CAMIF crop required by its active route.
  regs.push_back({0x050, sensor->bayer_pattern});
  regs.push_back({0xce4, sensor->frame_width - 1});
  regs.push_back({0xce8, sensor->frame_height - 1});
  // Preserve the module white balance; C4's sensor-side gains increased noise
  // in RAW10 captures. The rest of the ISP program is shared.
  regs.push_back({0x6fc, (0xbc << 16) | 0x80});
  regs.push_back({0x700, 0xd1});
  // Clear the extra interpolation slots initialized by the kernel baseline.
  for (uint32_t offset = 0x724; offset <= 0x75c; offset += 4) regs.push_back({offset, 0});
  if (!write_vfe_regs(regs, "upstream IFE")) return false;

  for (const auto &lut : luts) {
    const std::vector<uint32_t> *values = nullptr;
    switch (lut.bank) {
      case 9: values = &sensor->linearization_lut; break;
      case 14: case 15: values = &sensor->vignetting_lut; break;
      case 26: case 28: case 30: values = &sensor->gamma_lut_rgb; break;
      default: return false;
    }
    if (lut.bytes != values->size() * sizeof(uint32_t)) return false;
    VfeDmiCmd cmd = {};
    cmd.dmi_cfg_offset = lut.cfg_offset;
    cmd.ram_select = lut.bank;
    cmd.count = values->size();
    cmd.data = (uint64_t)(uintptr_t)values->data();
    if (v0_ioctl(video_fd, VFE_WRITE_DMI, &cmd) != 0) {
      LOGE("cam %d: failed upstream IFE LUT bank=%u: %d (%s)", cc.camera_num, lut.bank, errno, strerror(errno));
      return false;
    }
  }
  return v0_ioctl(video_fd, VFE_REG_UPDATE) == 0;
}

void OneCamera::camera_open(VisionIpcServer *v) {
  if (!enabled) return;

  vipc_server = v;
  stream_type = cc.stream_type;

  int cam_idx = cc.camera_num;
  auto &dcfg = v0_cams[cam_idx];
  sensor = std::make_unique<OS04C10>();
  LOG("cam %d: using OS04C10 RAW10 media path", cam_idx);
  sensor->bits_per_pixel = 10;
  sensor->black_level = 64;
  sensor->bayer_pattern = CAM_ISP_PATTERN_BAYER_RGRGRG;
  sensor->mipi_format = CAM_FORMAT_MIPI_RAW_10;
  sensor->frame_data_type = CSI_RAW10;
  sensor->frame_stride = sensor->frame_width * 10 / 8;
  // This RAW10 mode uses HTS=1070, half the qcom2 mode's 2140. Scale EV by
  // half as much so target-grey calculations represent the same exposure.
  sensor->ev_scale = 75.0f;
  sensor->exposure_time_max = 4717;
  sensor->max_ev = sensor->exposure_time_max * sensor->dc_gain_factor *
                   sensor->sensor_analog_gains[sensor->analog_gain_max_idx];

  if (dcfg.vfe_pix_entity == 0 || dcfg.pix_video_dev < 0 || dcfg.vfe_pix_subdev < 0) {
    LOGE("cam %d: required VFE PIX path is unavailable "
         "(entity=%u video=%d subdev=%d)",
         cam_idx, dcfg.vfe_pix_entity, dcfg.pix_video_dev, dcfg.vfe_pix_subdev);
    enabled = false;
    return;
  }

  const int output_scale = sensor->out_scale;
  output_width = std::max(2U, (sensor->frame_width / output_scale) & ~1U);
  output_height = std::max(2U, (sensor->frame_height / output_scale) & ~1U);
  auto [s, yh, uvh, sz] = get_nv12_info(output_width, output_height);
  vipc_stride = stride = s;
  vipc_y_height = y_height = yh;
  vipc_uv_height = uv_height = uvh;
  vipc_yuv_size = yuv_size = sz;
  vipc_uv_offset = uv_offset = stride * y_height;

  // open video device
  int dev = dcfg.pix_video_dev;
  std::string path = util::string_format("/dev/video%d", dev);
  video_fd = open(path.c_str(), O_RDWR);
  if (video_fd < 0) {
    LOGE("cam %d: failed to open %s: %d", cam_idx, path.c_str(), errno);
    enabled = false;
    return;
  }
  LOG("cam %d: opened %s (VFE PIX V4L2 DMABUF)", cam_idx, path.c_str());

  // find sensor subdev. The Dragon exposes enough CAMSS entities that the
  // third OS04 can land above v4l-subdev31.
  for (int i = 0; i < 96; i++) {
    const std::string name_path = util::string_format("/sys/class/video4linux/v4l-subdev%d/name", i);
    if (access(name_path.c_str(), R_OK) != 0) continue;
    std::string name = util::read_file(name_path);
    if (name.find(dcfg.sensor_name) == 0) {
      sensor_fd = open(util::string_format("/dev/v4l-subdev%d", i).c_str(), O_RDWR);
      break;
    }
  }
  if (sensor_fd < 0) {
    LOGE("cam %d: sensor subdev '%s' not found, disabling", cam_idx, dcfg.sensor_name);
    enabled = false;
    return;
  }

  if (!write_sensor_regs(sensor_fd, os04_default_init_regs(), "init file", cam_idx)) {
    enabled = false;
    return;
  }

  if (!apply_os04_20fps_timing(sensor_fd, cam_idx)) {
    enabled = false;
    return;
  }

  // Do not enable a CAMSS route until the sensor has responded. Leaving an
  // absent camera's media links enabled can interfere with working cameras.
  setup_media_links();
  set_formats();

  if (access("/dev/dma_heap/system", R_OK | W_OK) != 0) {
    LOGE("cam %d: required DMA heap is unavailable: %d (%s)", cam_idx, errno, strerror(errno));
    enabled = false;
    return;
  }

  // V4L2 buffers for normal VFE PIX NV12 frames.
  struct v4l2_requestbuffers req = {};
  req.count = n_bufs;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  req.memory = V4L2_MEMORY_DMABUF;
  int reqbufs_ret = ioctl(video_fd, VIDIOC_REQBUFS, &req);
  if (reqbufs_ret != 0) {
    LOGE("cam %d: REQBUFS DMABUF failed, disabling camera instead of using V4L2 MMAP CPU-copy path: %d (%s)",
         cam_idx, errno, strerror(errno));
    enabled = false;
    return;
  }
  n_bufs = req.count;

  // The Dragon VFE requires a wider native capture stride than the standard
  // Venus NV12 layout used by comma's VisionIPC consumers. Capture into a
  // private VFE ring, then publish normalized buffers below.
  vfe_buffers = std::make_unique<VisionBuf[]>(n_bufs);
  for (int i = 0; i < n_bufs; i++) {
    vfe_buffers[i].allocate(yuv_size);
  }

  v->create_buffers_with_sizes(stream_type, VIPC_BUFFER_COUNT,
                               output_width, output_height,
                               vipc_yuv_size, vipc_stride, vipc_uv_offset);

  LOG("cam %d: VIPC buffers created (%s, %ux%u, scale=%d, %u bytes, stride=%u; VFE stride=%u)",
      cam_idx, "VFE PIX V4L2 DMABUF NV12",
      output_width, output_height, output_scale, vipc_yuv_size, vipc_stride, stride);
}

void OneCamera::queue_all_buffers() {
  if (!enabled) return;

  for (int i = 0; i < n_bufs; i++) {
    queue_frame(i);
  }
}

void OneCamera::stream_on() {
  if (!enabled) return;

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (ioctl(video_fd, VIDIOC_STREAMON, &type) != 0) {
    LOGE("cam %d: STREAMON failed: %d (%s)", cc.camera_num, errno, strerror(errno));
    enabled = false;
    return;
  }

  if (!write_sensor_regs(sensor_fd, sensor->start_reg_array, "start", cc.camera_num)) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  if (!apply_vfe_tuning()) {
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
    enabled = false;
    return;
  }

  streaming = true;
  LOG("cam %d: VFE PIX V4L2 streaming started", cc.camera_num);
}

void OneCamera::stop_streaming() {
  if (streaming && sensor_fd >= 0) {
    write_sensor_regs(sensor_fd, {{0x100, 0}}, "stop", cc.camera_num);
  }

  if (video_fd >= 0 && streaming) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(video_fd, VIDIOC_STREAMOFF, &type);
  }
  streaming = false;
}

void OneCamera::queue_frame(int index) {
  struct v4l2_buffer vbuf = {};
  struct v4l2_plane planes[1] = {};
  vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vbuf.memory = V4L2_MEMORY_DMABUF;
  vbuf.index = index;
  vbuf.length = 1;
  vbuf.m.planes = planes;
  planes[0].m.fd = vfe_buffers[index].fd;
  planes[0].length = yuv_size;
  int ret = ioctl(video_fd, VIDIOC_QBUF, &vbuf);
  if (ret != 0) LOGE("cam %d: QBUF idx=%d failed: %d (%s)", cc.camera_num, index, errno, strerror(errno));
}

int OneCamera::dequeue_frame(uint64_t *timestamp) {
  struct pollfd pfd = {video_fd, POLLIN, 0};
  int ret = poll(&pfd, 1, 20);
  if (ret == 0) return -ETIMEDOUT;
  if (ret < 0) return -errno;
  if (!(pfd.revents & POLLIN)) return -EIO;

  struct v4l2_buffer dbuf = {};
  struct v4l2_plane planes[1] = {};
  dbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  dbuf.memory = V4L2_MEMORY_DMABUF;
  dbuf.length = 1;
  dbuf.m.planes = planes;
  if (ioctl(video_fd, VIDIOC_DQBUF, &dbuf) != 0) return -errno;

  *timestamp = (uint64_t)dbuf.timestamp.tv_sec * 1000000000ULL +
               (uint64_t)dbuf.timestamp.tv_usec * 1000ULL;
  return dbuf.index;
}

void OneCamera::set_exposure(int exposure_time, int gain_idx, bool dc_gain_enabled) {
  if (sensor_fd < 0) return;

  write_sensor_regs(sensor_fd, sensor->getExposureRegisters(exposure_time, gain_idx, dc_gain_enabled),
                    "exposure", cc.camera_num);
}

void OneCamera::camera_close() {
  if (video_fd >= 0) {
    stop_streaming();
    close(video_fd);
    video_fd = -1;
  }
  if (sensor_fd >= 0) {
    close(sensor_fd);
    sensor_fd = -1;
  }
  if (vfe_buffers != nullptr) {
    for (int i = 0; i < n_bufs; i++) {
      vfe_buffers[i].free();
    }
    vfe_buffers.reset();
  }
}

struct Os04AeSample {
  float grey_frac = 0.5f;
};

class CameraState : public AutoExposure {
public:
  OneCamera camera;
  uint64_t last_frame_ns = 0;
  Rect ae_xywh = {};

  uint32_t frame_id = 0;
  std::unique_ptr<PubMaster> pm;

  CameraState(const CameraConfig &config) : camera(config) {}
  ~CameraState() { camera.camera_close(); }

  void init(VisionIpcServer *v);
  void process_pix_frame(int buf_idx, uint64_t timestamp);
  void set_camera_exposure(const Os04AeSample &ae_sample);
  void set_exposure_rect();

};

void CameraState::init(VisionIpcServer *v) {
  camera.camera_open(v);
  if (!camera.enabled) return;

  // The v0 starts the road sensor first, then wide almost one period later.
  // Number wide one frame ahead so same-numbered road/wide frames refer to the
  // same 20 Hz capture instant.
  if (camera.cc.stream_type == VISION_STREAM_WIDE_ROAD) {
    frame_id = 1;
  }

  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});

  AutoExposure::init(camera.sensor.get(), camera.cc.camera_num, 600);
  camera.set_exposure(exposure_time, gain_idx, dc_gain_enabled);

  set_exposure_rect();
}

void CameraState::set_exposure_rect() {
  const float fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm / camera.sensor->out_scale;
  ae_xywh = get_exposure_rect(camera.cc.camera_num, fl_pix, camera.output_width, camera.output_height);
}

static Os04AeSample calculate_os04_ae_sample_nv12(const uint8_t *base, int stride, Rect ae_xywh,
                                                   int x_skip, int y_skip) {
  Os04AeSample ret;
  if (base == nullptr || stride <= 0) return ret;

  int lum_med;
  uint32_t lum_binning[256] = {0};

  unsigned int lum_total = 0;

  for (int y = ae_xywh.y; y < ae_xywh.y + ae_xywh.h; y += y_skip) {
    for (int x = ae_xywh.x; x < ae_xywh.x + ae_xywh.w; x += x_skip) {
      uint8_t lum = base[(y * stride) + x];
      lum_binning[lum]++;
      lum_total += 1;
    }
  }
  if (lum_total == 0) return ret;

  unsigned int lum_cur = 0;
  for (lum_med = 255; lum_med >= 0; lum_med--) {
    lum_cur += lum_binning[lum_med];
    if (lum_cur >= lum_total / 2) break;
  }

  ret.grey_frac = lum_med / 256.0f;
  return ret;
}

void CameraState::set_camera_exposure(const Os04AeSample &ae_sample) {
  if (!camera.enabled) return;
  const int old_exp_t = exposure_time;
  const int old_gain_idx = gain_idx;
  update(std::max(ae_sample.grey_frac, 1.0f / 256.0f), frame_id);
  if (exposure_time != old_exp_t || gain_idx != old_gain_idx) {
    camera.set_exposure(exposure_time, gain_idx, dc_gain_enabled);
  }
}

void CameraState::process_pix_frame(int buf_idx, uint64_t timestamp) {
  frame_id++;
  uint64_t timestamp_eof = timestamp + camera.sensor->readout_time_ns;

  VisionBuf *capture = &camera.vfe_buffers[buf_idx];
  VisionBuf *vb = camera.vipc_server->get_buffer(camera.stream_type, buf_idx);
  if (capture != nullptr && vb != nullptr) {
    capture->sync(VISIONBUF_SYNC_FROM_DEVICE);
    const uint8_t *nv12 = (const uint8_t *)capture->addr;
    set_camera_exposure(calculate_os04_ae_sample_nv12(nv12, camera.stride, ae_xywh, 2, camera.cc.stream_type != VISION_STREAM_CABIN ? 2 : 4));

    uint8_t *published = (uint8_t *)vb->addr;
    for (uint32_t y = 0; y < camera.output_height; y++) {
      memcpy(published + y * camera.vipc_stride, nv12 + y * camera.stride, camera.output_width);
    }
    for (uint32_t y = 0; y < camera.output_height / 2; y++) {
      memcpy(published + camera.vipc_uv_offset + y * camera.vipc_stride,
             nv12 + camera.uv_offset + y * camera.stride, camera.output_width);
    }
    vb->sync(VISIONBUF_SYNC_TO_DEVICE);
  }

  VisionIpcBufExtra extra = {frame_id, timestamp, timestamp_eof};
  vb->set_frame_id(frame_id);
  camera.vipc_server->send(vb, &extra, false);

  MessageBuilder msg;
  auto framed = (msg.initEvent().*camera.cc.init_camera_state)();
  framed.setFrameId(frame_id);
  framed.setRequestId(frame_id);
  framed.setTimestampEof(timestamp_eof);
  framed.setTimestampSof(timestamp);
  framed.setIntegLines(exposure_time);
  framed.setGain(camera.sensor->sensor_analog_gains[gain_idx] * get_gain_factor());
  framed.setHighConversionGain(dc_gain_enabled);
  framed.setSensor(camera.sensor->image_sensor);
  framed.setMeasuredGreyFraction(measured_grey_fraction);
  framed.setTargetGreyFraction(target_grey_fraction);
  framed.setExposureValPercent(util::map_val(cur_ev[frame_id % 3],
    camera.sensor->min_ev, camera.sensor->max_ev, 0.0f, 100.0f));
  pm->send(camera.cc.publish_name, msg);
}

void camerad_thread() {
  LOG("-- v0 camerad starting (VFE PIX DMABUF required; no runtime RDI/MMAP CPU fallback)");

  VisionIpcServer v("camerad");

  int media_fd = open("/dev/media0", O_RDWR);
  if (media_fd < 0) {
    LOGE("failed to open /dev/media0");
    return;
  }
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    if (!config.enabled) continue;

    const int i = config.camera_num;
    v0_cams[i] = resolve_cam_config(media_fd, i);
    LOG("cam %d: csiphy=%u csid=%u vfe_pix=%u pix_dev=%d pix_subdev=%d",
        i, v0_cams[i].csiphy_entity, v0_cams[i].csid_entity,
        v0_cams[i].vfe_pix_entity, v0_cams[i].pix_video_dev, v0_cams[i].vfe_pix_subdev);
  }
  close(media_fd);

  reset_all_media_links();

  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    if (!config.enabled) continue;

    auto cam = std::make_unique<CameraState>(config);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  for (auto &cam : cams) {
    cam->camera.queue_all_buffers();
  }
  constexpr int start_gap_us = 38000;
  auto stream_one = [&](const std::unique_ptr<CameraState> &cam) {
    cam->camera.stream_on();
    usleep(start_gap_us);
  };
  for (auto it = cams.rbegin(); it != cams.rend(); ++it) stream_one(*it);

  LOG("-- v0 camerad streaming");

  // Rebuild the complete media and VisionIPC graph when an expected camera
  // stops producing frames. VisionIPC streams cannot be recreated safely in
  // place, so returning lets manager restart camerad and re-probe every sensor.
  // While a flex is disconnected this repeats every thirty seconds; once it is
  // reconnected, the next camerad start restores the stream without a reboot.
  constexpr uint64_t camera_reconnect_timeout_ns = 30ULL * 1000 * 1000 * 1000;
  const uint64_t streaming_started_ns = nanos_since_boot();
  for (auto &cam : cams) cam->last_frame_ns = streaming_started_ns;

  while (!do_exit) {
    for (auto &cam : cams) {
      const uint64_t now = nanos_since_boot();
      if (!cam->camera.enabled || now - cam->last_frame_ns >= camera_reconnect_timeout_ns) {
        LOGE("cam %d: no frames for 30 seconds; restarting camerad to re-probe hardware",
             cam->camera.cc.camera_num);
        return;
      }

      uint64_t timestamp;
      int buf_idx = cam->camera.dequeue_frame(&timestamp);
      if (buf_idx < 0) {
        if (buf_idx != -ETIMEDOUT) {
          LOGW_100("cam %d: dequeue failed: %d (%s)", cam->camera.cc.camera_num, -buf_idx, strerror(-buf_idx));
          usleep(1000);
        }
        continue;
      }

      cam->last_frame_ns = nanos_since_boot();
      cam->process_pix_frame(buf_idx, timestamp);
      cam->camera.queue_frame(buf_idx);
    }
  }

  LOG("-- v0 camerad stopping");
  for (auto &cam : cams) {
    cam->camera.stop_streaming();
  }
}
