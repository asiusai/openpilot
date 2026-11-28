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

#include <time.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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

static constexpr uint32_t OS04_20FPS_VTS = 2270;

// Frame sync. One 20 Hz frame clock on the SoC is the reference, as the Panda timer is on comma four. Every sensor
// free-runs and a software phase lock (FrameClockLock) keeps it on the clock: road and wide at the tick, the driver
// camera half a period later. Hardware FSIN is not used: on the Dragon Q6A the road and wide FSIN lines need R358 and
// R121, and assembled devices missed 4 to 48% of the triggers through the camera board's 2.2 kOhm FSIN pull-down.
// CAMERAD_NO_FRAME_SYNC=1 turns the sync off: every sensor then free-runs at 20 FPS.
static bool frame_sync_enabled() {
  static const bool enabled = getenv("CAMERAD_NO_FRAME_SYNC") == nullptr;
  return enabled;
}
// Own frame with frame sync: a few lines shorter than 20 FPS. The lock then adds lines to every frame (3.6 on average
// for road and wide) and can move the sensor both ways. Exposure stays below this length.
static constexpr uint32_t OS04_SYNC_BASE_VTS = OS04_20FPS_VTS - 4;
static constexpr uint32_t OS04_DRIVER_VTS = 2360;  // comma's four-lane 20 FPS mode
static constexpr uint32_t OS04_DRIVER_SYNC_BASE_VTS = OS04_DRIVER_VTS - 4;
static uint32_t os04_road_vts() { return frame_sync_enabled() ? OS04_SYNC_BASE_VTS : OS04_20FPS_VTS; }
static uint32_t os04_driver_vts() { return frame_sync_enabled() ? OS04_DRIVER_SYNC_BASE_VTS : OS04_DRIVER_VTS; }

static const std::vector<i2c_random_wr_payload> &os04_default_init_regs(bool driver) {
  // Comma's documented 728 Mbps, four-lane, 20 FPS RAW12 mode.
  // Keep its PLL, MIPI and frame timing; use the existing Asius free-running
  // configuration because Dragon does not supply comma's external FSIN.
  static const std::vector<i2c_random_wr_payload> driver_regs = [] {
    std::vector<i2c_random_wr_payload> result(std::begin(init_array_os04c10), std::end(init_array_os04c10));
    // Free-run registers in place of comma's FSIN slave values.
    result.insert(result.end(), {
      {0x3002, 0x21}, {0x3663, 0x26}, {0x368a, 0x00}, {0x3822, 0x14},
      {0x3823, 0x08}, {0x3844, 0x08}, {0x382b, 0x00}, {0x3832, 0x00},
    });
    result.insert(result.end(), {
      // Match the Asius module orientation and Bayer crop used by the road cameras.
      // 0x3716 must match 0x3820's vertical orientation, as in os04c10_mirror_flip.
      {0x3716, 0x24},
      {0x3811, 0x09}, {0x3813, 0x09}, {0x3820, 0x88}, {0x3880, 0x25},
      {0x5100, 0x04}, {0x5101, 0x00}, {0x5140, 0x04}, {0x5141, 0x00},
      {0x5104, 0x04}, {0x5105, 0x00}, {0x5144, 0x04}, {0x5145, 0x00},
    });
    return result;
  }();
  static const std::vector<i2c_random_wr_payload> regs = [] {
    std::vector<i2c_random_wr_payload> result(std::begin(init_array_os04c10), std::end(init_array_os04c10));
    // Two-lane RAW12 mode: milkv-duo/duo-buildroot-sdk, 89f501af2a5c,
    // middleware/v2/component/isp/sensor/cv180x/ov_os04c10/os04c10_sensor_ctl.c.
    // Use the board's 24 MHz clock, 20 FPS timing, crop and VFE white balance.
    // Keep comma's shared initialization and apply only the mode differences.
    result.insert(result.end(), {
      {0x0100, 0x00}, {0x0305, 0x6a}, {0x0306, 0x00}, {0x0325, 0x54},
      {0x3016, 0x32}, {0x3106, 0x25}, {0x3501, 0x06}, {0x3502, 0x1e},
      {0x3512, 0x20}, {0x3660, 0x00}, {0x366a, 0x60}, {0x3698, 0x40},
      {0x369a, 0x18}, {0x369c, 0x14}, {0x36a1, 0x5d}, {0x370e, 0x0c},
      {0x3713, 0x00}, {0x3757, 0x0e}, {0x377b, 0x20}, {0x37be, 0x08},
      {0x37c7, 0x08}, {0x3832, 0x00}, {0x40ba, 0x00}, {0x4507, 0x64},
      {0x480e, 0x00}, {0x4813, 0x00}, {0x484b, 0x07}, {0x5036, 0x00},
      {0x301f, 0xd0}, {0x3610, 0x95}, {0x3681, 0xa6}, {0x3682, 0x53},
      {0x3683, 0x2a}, {0x3684, 0x15}, {0x370f, 0x04}, {0x3716, 0x24},
      {0x380e, 0x08}, {0x380f, 0xde}, {0x3811, 0x09}, {0x3813, 0x09},
      {0x3820, 0x88}, {0x3880, 0x25}, {0x4003, 0x80}, {0x4809, 0x1e},
      {0x4837, 0x12}, {0x3c8c, 0x10}, {0x37e0, 0x00}, {0x37e1, 0x04},
      {0x37e2, 0x08}, {0x37e4, 0x18}, {0x37e3, 0x04}, {0x3002, 0x21},
      {0x3663, 0x26}, {0x368a, 0x00}, {0x3822, 0x14}, {0x3823, 0x08},
      {0x3844, 0x08}, {0x382b, 0x00}, {0x5100, 0x04}, {0x5101, 0x00},
      {0x5140, 0x04}, {0x5141, 0x00}, {0x5104, 0x04}, {0x5105, 0x00},
      {0x5144, 0x04}, {0x5145, 0x00},
    });
    return result;
  }();
  return driver ? driver_regs : regs;
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

static bool apply_os04_20fps_timing(int sensor_fd, int cam_idx, bool driver) {
  const uint32_t vts = driver ? os04_driver_vts() : os04_road_vts();
  return write_sensor_regs(sensor_fd, {
    {0x380e, (vts >> 8) & 0xff},
    {0x380f, vts & 0xff},
  }, "20fps timing", cam_idx);
}

static int64_t raw_clock_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

// The 20 Hz frame clock. It runs on CLOCK_MONOTONIC_RAW: VamOS slews the other clocks by up to 500 ppm
// (25 us per frame), which a sensor oscillator cannot follow.
struct FrameClock {
  static constexpr int64_t PERIOD_NS = 50000000;
  std::atomic<int64_t> t0{0};  // time of one tick; 0 until started

  bool started() const { return t0.load() != 0; }
  // Signed distance of a time from the nearest tick, in (-P/2, P/2]. Positive: after the tick.
  int64_t phase(int64_t t_raw) const {
    int64_t e = (t_raw - t0.load()) % PERIOD_NS;
    if (e > PERIOD_NS / 2) e -= PERIOD_NS;
    if (e <= -PERIOD_NS / 2) e += PERIOD_NS;
    return e;
  }
};

// Software phase lock of one free-running sensor to the frame clock.
//
// The sensor's own frame (base VTS) is shorter than the clock period. After every frame the lock writes the VTS low
// byte 0x380f with a few more lines (about 22 us per line) for one frame:
//   - rate: the average number of lines that makes the sensor period equal to the clock period,
//     dithered between two integers and estimated from the phase change between windows
//   - phase: a part of the measured phase error, spread over the next window
// Start-up moves the sensor to the tick with one long frame (both VTS bytes), then confirms the phase.
// A write right after a frame's DMA completion (about line 1560 of the frame) applies to exactly one frame
// whether the sensor latches VTS at once or at the next frame. Frames never get shorter than the base frame,
// so exposure stays valid.
// Frame timestamps are noisy: on an idle board one in five is 0.75 ms late (CPU wake-up). Start-up therefore
// uses the earliest of a few samples, tracking uses a low order statistic of a window, and the loop is slow.
// The sensor oscillator itself drifts by ppm only.
class FrameClockLock {
public:
  // Sensor mode: base frame length (VTS), line time, and the lines per frame that match the clock at start.
  struct Mode { uint32_t base_vts; double line_ns; double rate; };
  static constexpr Mode ROAD_MODE = {OS04_SYNC_BASE_VTS, 22025.0, 3.6};       // measured: 49.996 ms / 2270 lines
  static constexpr Mode DRIVER_MODE = {OS04_DRIVER_SYNC_BASE_VTS, 21170.0, 7.0};  // measured: 49.962 ms / 2360 lines; 7.0 lines
  static constexpr int64_t TRACK_ENTER_NS = 150000;      // locked: phase within 0.15 ms at start
  static constexpr int64_t LOST_NS = 1500000;            // back to acquisition above 1.5 ms
  static constexpr int64_t ADVANCE_MAX_NS = 1500000;     // late by less: run short frames instead of a 50 ms delay
  static constexpr double JUMP_GATE_NS = 200000;         // phase jump between windows that the lines written do not explain

  int sensor_fd = -1;
  int cam_num = 0;
  const char *name = "";
  bool failed = false;
  bool locked = false;       // reached lock once; publishing may start
  double rate = ROAD_MODE.rate;  // lines per frame
  int64_t offset_ns = 0;     // target: the tick plus this offset

  void configure(const Mode &m, int64_t offset) {
    base_vts = m.base_vts;
    vts_lo = m.base_vts & 0xff;
    max_lines = 0xff - vts_lo;
    line_ns = m.line_ns;
    rate = m.rate;
    offset_ns = offset;
  }
  int64_t window_phase_ns = 0;
  int steps = 0, frames = 0;

  // Delay the sensor once by lengthening one frame. skip: frames whose phase is not valid after the request.
  void delay(int64_t ns, int skip) {
    delay_lines = (int)std::lround(ns / line_ns);
    settle = skip;
    acq_n = 0;
    steps++;
  }

  // Call after every frame. phase_ns: start of frame relative to the clock tick, positive when the sensor
  // is late. Before the clock runs (have_phase false) the lock only keeps the sensor at the clock rate.
  void on_frame(bool have_phase, int64_t phase_ns) {
    if (failed) return;
    frames++;
    if (have_phase) {
      if (tracking) track(phase_ns);
      else acquire(phase_ns);
    }

    // Lines for the next frame.
    int n = 0;
    if (advance_frames > 0) {
      advance_frames--;  // shortest frame: the sensor gains (rate * 22 us) on the clock
    } else {
      acc += rate + corr;
      n = std::clamp((int)std::floor(acc), 0, max_lines);
      acc = std::clamp(acc - n, -2.0, 2.0);
      lines_since_mid += n;  // delay frames are not part of the rate estimate: they happen during acquisition
    }
    frames_since_mid++;
    if (delay_lines > 0 || long_frame) {
      // One long frame, then back to the base frame: both VTS bytes in one write.
      const uint32_t vts = base_vts + n + delay_lines;
      long_frame = delay_lines > 0;
      delay_lines = 0;
      if (!write_sensor_regs(sensor_fd, {{0x380e, (vts >> 8) & 0xff}, {0x380f, vts & 0xff}}, "exposure", cam_num)) fail("VTS");
      written_lines = n;
    } else if (n != written_lines) {
      // The name "exposure" keeps write_sensor_regs quiet; these writes are per frame.
      if (!write_sensor_regs(sensor_fd, {{0x380f, (uint32_t)(vts_lo + n)}}, "exposure", cam_num)) fail("lines");
      written_lines = n;
    }
  }

private:
  static constexpr int MAX_WINDOW = 40;
  uint32_t base_vts = ROAD_MODE.base_vts, vts_lo = ROAD_MODE.base_vts & 0xff;
  int max_lines = 0xff - (ROAD_MODE.base_vts & 0xff);
  double line_ns = ROAD_MODE.line_ns;
  bool tracking = false;
  bool long_frame = false;
  int written_lines = 0;
  double acc = 0.0, corr = 0.0;
  int delay_lines = 0, advance_frames = 0, settle = 0;
  int64_t acq_min = 0;
  int acq_n = 0;
  int64_t window[MAX_WINDOW] = {};
  int window_n = 0, windows = 0, gated = 0;
  bool have_prev = false;
  double prev_phase_lines = 0.0;
  int lines_since_mid = 0, frames_since_mid = 0;   // since the middle of the previous window
  int lines_half = 0, frames_half = 0;             // the same counters at the middle of the current window

  // Window length in frames, phase gain, rate gain, largest phase step per window in lines, largest rate step.
  struct Gains { int window; double phase, rate, max_lines, max_rate_step; };
  static Gains gains(int windows_done) {
    if (windows_done < 3) return {8, 0.6, 0.6, 40.0, 2.0};     // 0.4 s windows: pull in
    if (windows_done < 13) return {20, 0.5, 0.4, 12.0, 0.15};  // 1 s windows: settle the rate
    return {MAX_WINDOW, 0.3, 0.2, 4.0, 0.03};                  // 2 s windows: hold
  }

  void fail(const char *what) {
    LOGE("frame clock: %s %s write failed; lock disabled", name, what);
    failed = true;
  }

  void acquire(int64_t phase_ns) {
    if (settle > 0) { settle--; return; }
    // A late timestamp only makes the phase look later: the earliest of the samples is the phase.
    acq_min = acq_n == 0 ? phase_ns : std::min(acq_min, phase_ns);
    acq_n++;
    window_phase_ns = acq_min;
    if (acq_n < 2) return;
    if (std::llabs(acq_min) <= TRACK_ENTER_NS) {
      tracking = true;
      window_n = windows = gated = 0;
      have_prev = false;
      corr = 0.0;
      if (!locked) {
        locked = true;
        LOGW("frame clock: %s locked, phase %.3f ms after %d frames, %d steps", name, acq_min / 1e6, frames, steps);
      }
      return;
    }
    if (acq_n < 4) return;  // not in tolerance: look at two more frames before a step
    if (acq_min > 0 && acq_min <= ADVANCE_MAX_NS) {
      advance_frames = std::max(1, (int)std::lround(acq_min / (rate * line_ns)));
      settle = advance_frames + 1;
      acq_n = 0;
      steps++;
      return;
    }
    delay(acq_min < 0 ? -acq_min : FrameClock::PERIOD_NS - acq_min, 1);
  }

  void track(int64_t phase_ns) {
    const Gains g = gains(windows);
    window[window_n++] = phase_ns;
    if (window_n == g.window / 2) {
      lines_half = lines_since_mid;
      frames_half = frames_since_mid;
    }
    if (window_n < g.window) return;
    // Late timestamps only move the upper part of the distribution: use a low order statistic.
    std::nth_element(window, window + g.window / 5, window + g.window);
    const int64_t m = window[g.window / 5];
    window_phase_ns = m;
    window_n = 0;
    windows++;
    if (std::llabs(m) > LOST_NS) {
      LOGW("frame clock: %s lost the clock, phase %.3f ms; acquiring again", name, m / 1e6);
      tracking = false;
      acq_n = 0;
      return;
    }
    const double phase_lines = m / line_ns;
    if (have_prev && frames_half > 0) {
      // Between the two window middles the phase changed by (lines added - rate needed), frame by frame.
      const double rate_est = (lines_half - (phase_lines - prev_phase_lines)) / frames_half;
      // A sensor cannot jump. A window that disagrees with the lines written is a burst of late timestamps:
      // ignore up to two such windows in a row and keep the rate.
      const double jump_ns = (rate - rate_est) * frames_half * line_ns;
      if (windows > 3 && std::abs(jump_ns) > JUMP_GATE_NS && gated < 2) {
        gated++;
        corr = 0.0;
        LOGD("frame clock: %s window ignored, phase %.3f ms, jump %.3f ms", name, m / 1e6, jump_ns / 1e6);
        return;
      }
      rate = std::clamp(rate + std::clamp(g.rate * (rate_est - rate), -g.max_rate_step, g.max_rate_step), 0.5, 30.0);
    }
    gated = 0;
    have_prev = true;
    prev_phase_lines = phase_lines;
    lines_since_mid -= lines_half;
    frames_since_mid -= frames_half;
    corr = -std::clamp(g.phase * phase_lines, -g.max_lines, g.max_lines) / gains(windows).window;
    LOGD("frame clock: %s phase %.3f ms rate %.3f lines", name, m / 1e6, rate);
  }
};

// Starts the frame clock in phase with the later of road and wide, locks both to the tick and the driver camera to
// the tick plus half a period.
class FrameSync {
public:
  static constexpr int ROAD = 0, WIDE = 1;
  static constexpr int START_SAMPLES = 2;               // frames of each sensor measured before the clock starts
  static constexpr int ACQUIRE_TIMEOUT_FRAMES = 160;    // 8 s, then publish unlocked

  bool enabled = false;
  bool have_driver = false;
  bool timed_out = false;
  FrameClock clock;
  FrameClockLock cam[2];
  FrameClockLock driver;

  // Road and wide frames are held back while this is true (see camerad_thread). They wait for the driver lock too,
  // so that all three cameras start publishing with the same frame ID.
  bool acquiring() const {
    if (!enabled || timed_out || cam[ROAD].failed || cam[WIDE].failed) return false;
    const bool driver_ready = !have_driver || driver.locked || driver.failed;
    return !(cam[ROAD].locked && cam[WIDE].locked && driver_ready);
  }

  void on_frame(int which, int64_t sof_raw_ns) {
    if (!enabled) return;
    frames++;
    const bool started = clock.started();
    if (!started) {
      // Both sensors already run at the clock rate. The earliest sample, moved to this frame, is the sensor's phase.
      const int64_t P = FrameClock::PERIOD_NS;
      sof[which] = seen[which] ? std::min(sof[which] + P, sof_raw_ns) : sof_raw_ns;
      seen[which]++;
      if (seen[ROAD] >= START_SAMPLES && seen[WIDE] >= START_SAMPLES) {
        // Road start of frame minus wide start of frame, wrapped. The later sensor defines the tick; the other is delayed.
        int64_t d = (sof[ROAD] - sof[WIDE]) % P;
        if (d > P / 2) d -= P;
        if (d <= -P / 2) d += P;
        const int later = d >= 0 ? ROAD : WIDE;
        clock.t0.store(sof[later]);
        // The long frame is written at that sensor's next call: now for this sensor, one frame later for the other.
        cam[1 - later].delay(std::llabs(d), 1 - later == which ? 1 : 2);
        LOGW("frame clock started on the %s camera; road minus wide %.3f ms", later == ROAD ? "road" : "wide", d / 1e6);
      }
    }
    cam[which].on_frame(started, started ? clock.phase(sof_raw_ns) : 0);
    if (!timed_out && acquiring() && frames >= 2 * ACQUIRE_TIMEOUT_FRAMES) {
      timed_out = true;
      LOGE("frame clock: lock timed out, road phase %.3f ms, wide phase %.3f ms, driver phase %.3f ms; publishing unlocked",
           cam[ROAD].window_phase_ns / 1e6, cam[WIDE].window_phase_ns / 1e6, driver.window_phase_ns / 1e6);
    }
  }

  // Driver start of frame against the tick plus half a period, or 0 before the clock runs.
  int64_t on_driver_frame(int64_t sof_raw_ns) {
    if (!enabled || !have_driver) return 0;
    const bool started = clock.started();
    const int64_t phase = started ? clock.phase(sof_raw_ns - driver.offset_ns) : 0;
    driver.on_frame(started, phase);
    return phase;
  }

private:
  int frames = 0;
  int seen[2] = {};
  int64_t sof[2] = {};
};

class OneCamera {
public:
  CameraConfig cc;
  std::unique_ptr<SensorInfo> sensor;
  bool enabled;

  int video_fd = -1;
  int sensor_fd = -1;
  bool streaming = false;
  uint64_t last_capture_timestamp = 0;

  int n_bufs = 4;
  std::unique_ptr<VisionBuf[]> vfe_buffers;

  uint32_t output_width = 0, output_height = 0;
  uint32_t stride = 0, y_height = 0, yuv_size = 0, uv_offset = 0;
  uint32_t vipc_stride = 0, vipc_yuv_size = 0, vipc_uv_offset = 0;

  OneCamera(const CameraConfig &config) : cc(config), enabled(config.enabled) {}

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void setup_media_links();
  bool set_formats();
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

bool OneCamera::set_formats() {
  int cam_idx = cc.camera_num;
  auto &dcfg = v0_cams[cam_idx];
  const uint32_t media_bus_code = MEDIA_BUS_FMT_SBGGR12_1X12;
  bool valid = true;
  const auto set_format = [cam_idx](int fd, v4l2_subdev_format &fmt, const char *stage) {
    const auto requested = fmt.format;
    if (ioctl(fd, VIDIOC_SUBDEV_S_FMT, &fmt) != 0) {
      LOGE("cam %d: %s pad %u S_FMT failed: %d (%s)", cam_idx, stage, fmt.pad, errno, strerror(errno));
      return false;
    }
    if (fmt.format.code != requested.code || fmt.format.width != requested.width || fmt.format.height != requested.height) {
      LOGE("cam %d: %s pad %u rejected required format %ux%u code=0x%x; got %ux%u code=0x%x",
           cam_idx, stage, fmt.pad, requested.width, requested.height, requested.code,
           fmt.format.width, fmt.format.height, fmt.format.code);
      return false;
    }
    return true;
  };
  // set format on CSIPHY subdev
  int csiphy_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.csiphy_subdev).c_str(), O_RDWR);
  if (csiphy_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    valid &= set_format(csiphy_fd, sfmt, "CSIPHY");
    sfmt.pad = 1;
    valid &= set_format(csiphy_fd, sfmt, "CSIPHY");
    close(csiphy_fd);
  } else {
    return false;
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
    valid &= set_format(csid_fd, sfmt, "CSID");

    // PIX source pad matching the selected virtual channel.
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = ONE_PIX_CSID_SOURCE_PAD;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    valid &= set_format(csid_fd, sfmt, "CSID");
    close(csid_fd);
  } else {
    return false;
  }

  // set format on sensor subdev
  if (sensor_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    valid &= set_format(sensor_fd, sfmt, "sensor");
  } else {
    return false;
  }

  int vfe_pix_fd = open(util::string_format("/dev/v4l-subdev%d", dcfg.vfe_pix_subdev).c_str(), O_RDWR);
  if (vfe_pix_fd >= 0) {
    struct v4l2_subdev_format sfmt = {};
    sfmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sfmt.pad = 0;
    sfmt.format.width = sensor->frame_width;
    sfmt.format.height = sensor->frame_height;
    sfmt.format.code = media_bus_code;
    valid &= set_format(vfe_pix_fd, sfmt, "VFE PIX");

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
      valid = false;
    }

    sfmt.pad = 1;
    sfmt.format.width = output_width;
    sfmt.format.height = output_height;
    sfmt.format.code = MEDIA_BUS_FMT_YUYV8_1_5X8;
    valid &= set_format(vfe_pix_fd, sfmt, "VFE PIX");
    close(vfe_pix_fd);
  } else {
    return false;
  }

  if (!valid) return false;

  struct v4l2_format vfmt = {};
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  vfmt.fmt.pix_mp.width = output_width;
  vfmt.fmt.pix_mp.height = output_height;
  vfmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  vfmt.fmt.pix_mp.num_planes = 1;
  vfmt.fmt.pix_mp.plane_fmt[0].bytesperline = stride;
  vfmt.fmt.pix_mp.plane_fmt[0].sizeimage = yuv_size;
  if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) == 0) {
    if (vfmt.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 || vfmt.fmt.pix_mp.num_planes != 1 ||
        vfmt.fmt.pix_mp.width != output_width || vfmt.fmt.pix_mp.height != output_height) {
      LOGE("cam %d: VFE PIX rejected required NV12 output", cam_idx);
      return false;
    }
    stride = vfmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    y_height = output_height;
    uv_offset = stride * y_height;
    yuv_size = vfmt.fmt.pix_mp.plane_fmt[0].sizeimage;
    LOG("cam %d: VFE PIX format set raw=%dx%d out=%dx%d stride=%u size=%u video_stride=%u sizeimage=%u",
        cam_idx, sensor->frame_width, sensor->frame_height, vfmt.fmt.pix_mp.width, vfmt.fmt.pix_mp.height,
        stride, yuv_size, vfmt.fmt.pix_mp.plane_fmt[0].bytesperline, vfmt.fmt.pix_mp.plane_fmt[0].sizeimage);
  } else {
    LOGE("cam %d: VFE PIX S_FMT failed: %d (%s)", cam_idx, errno, strerror(errno));
    return false;
  }
  return true;
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
  // Apply module white balance in the ISP. Keep unity green gain so clipped
  // highlights remain neutral. The rest of the ISP program is shared.
  regs.push_back({0x6fc, (0xc6 << 16) | 0x80});  // blue, green
  regs.push_back({0x700, 0xe8});                 // red
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
  const bool driver = cam_idx == 2;
  const int lanes = driver ? 4 : 2;
  const int bus = driver ? 16 : (cam_idx == 1 ? 18 : 20);
  const auto lane_config = util::read_file(util::string_format(
      "/sys/bus/i2c/devices/%d-0036/of_node/port/endpoint/data-lanes", bus));
  if (lane_config.size() != lanes * sizeof(uint32_t)) {
    LOGE("cam %d: requires %d CSI-2 lanes; update the device kernel and device tree", cam_idx, lanes);
    enabled = false;
    return;
  }
  sensor = std::make_unique<OS04C10>();
  sensor->bits_per_pixel = 12;
  LOG("cam %d: using OS04C10 RAW%d media path, %d CSI-2 lanes", cam_idx, sensor->bits_per_pixel, lanes);
  // The four-lane mode retains the sensor's upstream black level.
  // Only the two-lane table changes its black-level target to 128.
  if (!driver) sensor->black_level = 128;
  sensor->bayer_pattern = CAM_ISP_PATTERN_BAYER_RGRGRG;
  sensor->mipi_format = CAM_FORMAT_MIPI_RAW_12;
  sensor->frame_data_type = CSI_RAW12;
  sensor->frame_stride = sensor->frame_width * sensor->bits_per_pixel / 8;
  // RAW12: HTS=2140; PLL2=354 for driver and 340 for road cameras.
  // Scale EV by the exposure-line duration to retain the same metering policy.
  sensor->ev_scale = driver ? 150.0f : 150.0f * 354.0f / 340.0f;
  sensor->exposure_time_max = (driver ? os04_driver_vts() : os04_road_vts()) - 8;
  // Measured CSID SOF-to-EOF intervals for the two sensor modes.
  sensor->readout_time_ns = driver ? 32934000 : 34234000;
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
  auto [s, yh, _, sz] = get_nv12_info(output_width, output_height);
  vipc_stride = stride = s;
  y_height = yh;
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

  if (!write_sensor_regs(sensor_fd, os04_default_init_regs(driver), "init file", cam_idx)) {
    enabled = false;
    return;
  }

  if (!apply_os04_20fps_timing(sensor_fd, cam_idx, driver)) {
    enabled = false;
    return;
  }

  // Do not enable a CAMSS route until the sensor has responded. Leaving an
  // absent camera's media links enabled can interfere with working cameras.
  setup_media_links();
  if (!set_formats()) {
    LOGE("cam %d: required RAW12 to NV12 media path is unavailable", cam_idx);
    enabled = false;
    return;
  }

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
  int ret = poll(&pfd, 1, 0);
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
  // Use capture timestamps: the VFE sequence only counts completed buffers,
  // so a frame lost before DMA completion does not leave a sequence gap.
  if (last_capture_timestamp && (*timestamp <= last_capture_timestamp || *timestamp - last_capture_timestamp > 75000000ULL)) {
    LOGW("cam %d: capture gap %.2f ms, dequeue delay %.2f ms, sequence=%u flags=0x%x",
         cc.camera_num, (int64_t)(*timestamp - last_capture_timestamp) / 1e6,
         (int64_t)(nanos_since_boot() - *timestamp) / 1e6, dbuf.sequence, dbuf.flags);
  }
  last_capture_timestamp = *timestamp;
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
  uint64_t last_capture_eof = 0;
  Rect ae_xywh = {};

  uint32_t frame_id = 0;
  // False while the road/wide frame-phase lock acquires; frames then only feed exposure.
  bool publish = true;
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

  AutoExposure::init(camera.sensor.get(), camera.cc.camera_num, 200);
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
  // Mainline VFE timestamps DMA completion. Estimate SOF from the readout time.
  const uint64_t timestamp_eof = timestamp;
  timestamp -= camera.sensor->readout_time_ns;

  VisionBuf *capture = &camera.vfe_buffers[buf_idx];
  VisionBuf *vb = camera.vipc_server->get_buffer(camera.stream_type, buf_idx);
  if (capture != nullptr && vb != nullptr) {
    capture->sync(VISIONBUF_SYNC_FROM_DEVICE);
    const uint8_t *nv12 = (const uint8_t *)capture->addr;
    set_camera_exposure(calculate_os04_ae_sample_nv12(nv12, camera.stride, ae_xywh, 2, camera.cc.stream_type != VISION_STREAM_CABIN ? 2 : 4));
    if (!publish) return;

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

  if (!publish) return;
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

  // Frame sync (see frame_sync_enabled). Set up after stream_on, which disables a camera that fails to start.
  FrameSync sync;
  CameraState *lock_cam[2] = {nullptr, nullptr};
  CameraState *driver_cam = nullptr;
  for (auto &cam : cams) {
    if (!cam->camera.enabled) continue;
    if (cam->camera.cc.stream_type == VISION_STREAM_NARROW_ROAD) lock_cam[FrameSync::ROAD] = cam.get();
    if (cam->camera.cc.stream_type == VISION_STREAM_WIDE_ROAD) lock_cam[FrameSync::WIDE] = cam.get();
    if (cam->camera.cc.stream_type == VISION_STREAM_CABIN) driver_cam = cam.get();
  }
  sync.enabled = lock_cam[0] && lock_cam[1] && frame_sync_enabled();
  if (sync.enabled) {
    for (int k = 0; k < 2; k++) {
      sync.cam[k].configure(FrameClockLock::ROAD_MODE, 0);
      sync.cam[k].sensor_fd = lock_cam[k]->camera.sensor_fd;
      sync.cam[k].cam_num = lock_cam[k]->camera.cc.camera_num;
      sync.cam[k].name = k == FrameSync::ROAD ? "road" : "wide";
      lock_cam[k]->publish = false;
    }
    if (driver_cam != nullptr) {
      // The driver camera locks to the tick plus half a period. Its frames are held back until it is locked.
      sync.have_driver = true;
      sync.driver.configure(FrameClockLock::DRIVER_MODE, FrameClock::PERIOD_NS / 2);
      sync.driver.sensor_fd = driver_cam->camera.sensor_fd;
      sync.driver.cam_num = driver_cam->camera.cc.camera_num;
      sync.driver.name = "driver";
      driver_cam->publish = false;
    }
    LOG("frame clock lock enabled for road, wide%s", driver_cam != nullptr ? " and driver" : "");
  }
  int64_t driver_phase[40] = {};
  int driver_phase_n = 0, driver_held = 0;
  bool pair_started = false;
  uint32_t pair_first_id = 0;
  bool driver_reported = false;

  // Rebuild the complete media and VisionIPC graph when an expected camera
  // stops producing frames. VisionIPC streams cannot be recreated safely in
  // place, so returning lets manager restart camerad and re-probe every sensor.
  // While a flex is disconnected this repeats every thirty seconds; once it is
  // reconnected, the next camerad start restores the stream without a reboot.
  constexpr uint64_t camera_reconnect_timeout_ns = 30ULL * 1000 * 1000 * 1000;
  const uint64_t streaming_started_ns = nanos_since_boot();
  for (auto &cam : cams) cam->last_frame_ns = streaming_started_ns;

  std::vector<pollfd> poll_fds;
  for (const auto &cam : cams) {
    poll_fds.push_back({cam->camera.enabled ? cam->camera.video_fd : -1, POLLIN, 0});
  }

  while (!do_exit) {
    // Wait for whichever camera finishes first, without delaying ready frames
    // while another camera is still reading out its sensor.
    if (poll(poll_fds.data(), poll_fds.size(), 20) < 0) {
      if (errno != EINTR) LOGE("camera poll failed: %d (%s)", errno, strerror(errno));
      continue;
    }
    for (size_t i = 0; i < cams.size(); i++) {
      auto &cam = cams[i];
      const uint64_t now = nanos_since_boot();
      if (now - cam->last_frame_ns >= camera_reconnect_timeout_ns) {
        LOGE("cam %d: no frames for 30 seconds; restarting camerad to re-probe hardware",
             cam->camera.cc.camera_num);
        return;
      }
      if (!cam->camera.enabled || !(poll_fds[i].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL))) continue;

      uint64_t timestamp;
      int buf_idx = cam->camera.dequeue_frame(&timestamp);
      if (buf_idx < 0) {
        if (buf_idx != -ETIMEDOUT) {
          cloudlog_rl(2, 100, CLOUDLOG_WARNING, "cam %d: dequeue failed: %d (%s)", cam->camera.cc.camera_num, -buf_idx, strerror(-buf_idx));
          usleep(1000);
        }
        continue;
      }

      cam->last_frame_ns = nanos_since_boot();
      // Frame timestamps are on the slewed clock. Move them to the raw clock of the frame clock.
      const int64_t sof_raw_ns = raw_clock_ns() - ((int64_t)cam->last_frame_ns - (int64_t)timestamp) -
                                 (int64_t)cam->camera.sensor->readout_time_ns;
      const int lock_idx = cam.get() == lock_cam[0] ? 0 : (cam.get() == lock_cam[1] ? 1 : -1);
      bool publish_after_frame = false;
      if (lock_idx >= 0 && sync.enabled) {
        sync.on_frame(lock_idx, sof_raw_ns);
        // The driver lock can complete between road and wide frames, so test the state, not a change.
        if (!pair_started && !sync.acquiring()) {
          // Start publishing with a pair: two captures within half a period get the same frame ID.
          CameraState *other = lock_cam[1 - lock_idx];
          const bool other_delivered = std::llabs((int64_t)timestamp - (int64_t)other->last_capture_eof) < FrameClock::PERIOD_NS / 2;
          // Frame ID of this frame's pair. With a driver camera it is the ID of the driver frame of the same tick,
          // as on comma devices. That frame ends later: half a period, less the shorter driver readout.
          uint32_t pair_id = other_delivered ? other->frame_id : cam->frame_id + 1;
          if (driver_cam != nullptr && driver_cam->last_capture_eof != 0) {
            const int64_t lag = FrameClock::PERIOD_NS / 2 + (int64_t)driver_cam->camera.sensor->readout_time_ns -
                                (int64_t)cam->camera.sensor->readout_time_ns;
            const int64_t since_driver = (int64_t)timestamp - (int64_t)driver_cam->last_capture_eof;
            pair_id = driver_cam->frame_id + (uint32_t)std::llround((double)(since_driver + lag) / FrameClock::PERIOD_NS);
          }
          cam->frame_id = pair_id - 1;  // process_pix_frame increments it
          if (other_delivered) {
            // The other camera already delivered this pair unpublished: start with the next pair.
            other->frame_id = pair_id;
            publish_after_frame = true;
          } else {
            // The other camera's next frame is this frame's pair.
            other->frame_id = pair_id - 1;
            lock_cam[0]->publish = lock_cam[1]->publish = true;
          }
          pair_started = true;
          pair_first_id = pair_id + publish_after_frame;
          LOG("frame clock: publishing road and wide from frame %u", pair_first_id);
        }
      } else if (cam.get() == driver_cam && sync.enabled) {
        const int64_t phase = sync.on_driver_frame(sof_raw_ns);
        if (!cam->publish) {
          // Publish once the driver lock holds and road and wide have started; all three then share frame IDs.
          const bool ready = sync.driver.locked && pair_started && cam->frame_id + 1 >= pair_first_id;
          if (ready || sync.driver.failed || ++driver_held >= 2 * FrameSync::ACQUIRE_TIMEOUT_FRAMES) {
            cam->publish = true;
            if (ready) LOG("frame clock: driver camera locked; publishing from frame %u", cam->frame_id + 1);
            else LOGE("frame clock: driver camera not locked, phase %.3f ms; publishing", phase / 1e6);
          }
        }
        if (sync.clock.started()) {
          // Report the phase once and when it moves.
          driver_phase[driver_phase_n++] = phase;
          if (driver_phase_n == 40) {
            driver_phase_n = 0;
            std::nth_element(driver_phase, driver_phase + 8, driver_phase + 40);
            const int64_t m = driver_phase[8];
            if (!driver_reported || std::llabs(m) > 500000) {
              cloudlog_rl(4, 60000, std::llabs(m) > 500000 ? CLOUDLOG_WARNING : CLOUDLOG_INFO,
                          "frame clock: driver camera %.3f ms from the tick plus half a period", m / 1e6);
              driver_reported = true;
            }
            LOGD("frame clock: driver phase %.3f ms", m / 1e6);
          }
        }
      }
      if (sync.enabled && cam->publish && cam->last_capture_eof != 0) {
        const int64_t periods = std::llround((double)((int64_t)timestamp - (int64_t)cam->last_capture_eof) / FrameClock::PERIOD_NS);
        if (periods > 1 && periods < 100) {
          cam->frame_id += (uint32_t)(periods - 1);
          cloudlog_rl(4, 10000, CLOUDLOG_WARNING, "cam %d: %d frame(s) lost", cam->camera.cc.camera_num, (int)(periods - 1));
        }
      }
      cam->last_capture_eof = timestamp;
      cam->process_pix_frame(buf_idx, timestamp);
      if (publish_after_frame) lock_cam[0]->publish = lock_cam[1]->publish = true;
      cam->camera.queue_frame(buf_idx);
    }
  }

  LOG("-- v0 camerad stopping");
  for (auto &cam : cams) {
    cam->camera.stop_streaming();
  }
}
