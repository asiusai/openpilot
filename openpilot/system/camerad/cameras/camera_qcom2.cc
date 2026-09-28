#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/spectra.h"
#include "system/camerad/cameras/autoexposure.h"

#include <poll.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "common/params.h"
#include "common/swaglog.h"


ExitHandler do_exit;

// for debugging
const bool env_debug_frames = getenv("DEBUG_FRAMES") != nullptr;
const bool env_log_raw_frames = getenv("LOG_RAW_FRAMES") != nullptr;
const bool env_ctrl_exp_from_params = getenv("CTRL_EXP_FROM_PARAMS") != nullptr;


class CameraState : public AutoExposure {
public:
  SpectraCamera camera;
  Rect ae_xywh = {};

  float fl_pix = 0;
  std::unique_ptr<PubMaster> pm;

  CameraState(SpectraMaster *master, const CameraConfig &config) : camera(master, config) {};
  ~CameraState();
  void init(VisionIpcServer *v);
  void set_camera_exposure(float grey_frac);
  void set_exposure_rect();
  void sendState();
};

void CameraState::init(VisionIpcServer *v) {
  camera.camera_open(v);

  if (!camera.enabled) return;

  fl_pix = camera.cc.focal_len / camera.sensor->pixel_size_mm / camera.sensor->out_scale;
  set_exposure_rect();

  AutoExposure::init(camera.sensor.get(), camera.cc.camera_num);

  pm = std::make_unique<PubMaster>(std::vector{camera.cc.publish_name});
}

CameraState::~CameraState() {}

void CameraState::set_exposure_rect() {
  ae_xywh = get_exposure_rect(camera.cc.camera_num, fl_pix, camera.buf.out_img_width, camera.buf.out_img_height);
}

void CameraState::set_camera_exposure(float grey_frac) {
  if (!camera.enabled) return;
  int override_gain = -1, override_time = -1;
  if (env_ctrl_exp_from_params) {
    static Params params;
    auto gain_bytes = params.get("CameraDebugExpGain");
    auto time_bytes = params.get("CameraDebugExpTime");
    if (!gain_bytes.empty() && !time_bytes.empty()) {
      override_gain = std::stoi(gain_bytes);
      override_time = std::stoi(time_bytes);
    }
  }
  update(grey_frac, camera.buf.cur_frame_data.frame_id, override_gain, override_time);
  auto regs = camera.sensor->getExposureRegisters(exposure_time, gain_idx, dc_gain_enabled);
  camera.sensors_i2c(regs.data(), regs.size(), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, camera.sensor->data_word);
}

void CameraState::sendState() {
  camera.buf.sendFrameToVipc();

  MessageBuilder msg;
  auto framed = (msg.initEvent().*camera.cc.init_camera_state)();
  const FrameMetadata &meta = camera.buf.cur_frame_data;
  framed.setFrameId(meta.frame_id);
  framed.setRequestId(meta.request_id);
  framed.setTimestampEof(meta.timestamp_eof);
  framed.setTimestampSof(meta.timestamp_sof);
  framed.setIntegLines(exposure_time);
  framed.setGain(analog_gain_frac * get_gain_factor());
  framed.setHighConversionGain(dc_gain_enabled);
  framed.setMeasuredGreyFraction(measured_grey_fraction);
  framed.setTargetGreyFraction(target_grey_fraction);
  framed.setProcessingTime(meta.processing_time);

  const float ev = cur_ev[meta.frame_id % 3];
  const float perc = util::map_val(ev, camera.sensor->min_ev, camera.sensor->max_ev, 0.0f, 100.0f);
  framed.setExposureValPercent(perc);
  framed.setSensor(camera.sensor->image_sensor);

  // Log raw frames for road camera
  if (env_log_raw_frames && camera.cc.stream_type == VISION_STREAM_NARROW_ROAD && meta.frame_id % 100 == 5) {  // no overlap with qlog decimation
    framed.setImage(get_raw_frame_image(&camera.buf));
  }

  set_camera_exposure(calculate_exposure_value(&camera.buf, ae_xywh, 2, camera.cc.stream_type != VISION_STREAM_CABIN ? 2 : 4));

  // Send the message
  pm->send(camera.cc.publish_name, msg);
}

void camerad_thread() {
  // TODO: centralize enabled handling

  VisionIpcServer v("camerad");

  // *** initial ISP init ***
  SpectraMaster m;
  m.init();

  // *** per-cam init ***
  std::vector<std::unique_ptr<CameraState>> cams;
  for (const auto &config : ALL_CAMERA_CONFIGS) {
    auto cam = std::make_unique<CameraState>(&m, config);
    cam->init(&v);
    cams.emplace_back(std::move(cam));
  }

  v.start_listener();

  // start devices
  LOG("-- Starting devices");
  for (auto &cam : cams) cam->camera.sensors_start();

  // poll events
  LOG("-- Dequeueing Video events");
  while (!do_exit) {
    struct pollfd fds[1] = {{.fd = m.video0_fd, .events = POLLPRI}};
    int ret = poll(fds, std::size(fds), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    if (!(fds[0].revents & POLLPRI)) continue;

    struct v4l2_event ev = {0};
    ret = HANDLE_EINTR(ioctl(fds[0].fd, VIDIOC_DQEVENT, &ev));
    if (ret == 0) {
      if (ev.type == V4L_EVENT_CAM_REQ_MGR_EVENT) {
        struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)ev.u.data;
        if (env_debug_frames) {
          printf("sess_hdl 0x%6X, link_hdl 0x%6X, frame_id %lu, req_id %lu, timestamp %.2f ms, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl,
                 event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp/1e6, event_data->u.frame_msg.sof_status);
          do_exit = do_exit || event_data->u.frame_msg.frame_id > (1*20);
        }

        for (auto &cam : cams) {
          if (event_data->session_hdl == cam->camera.session_handle) {
            if (cam->camera.handle_camera_event(event_data)) {
              cam->sendState();
            }
            break;
          }
        }
      } else {
        LOGE("unhandled event %d\n", ev.type);
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }
}
