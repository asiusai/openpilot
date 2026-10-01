#include <array>
#include <cmath>
#include <map>

#include "common/tests/native_test.h"
#include "system/camerad/cameras/autoexposure.h"
#include "system/camerad/cameras/cdm_parse.h"
#include "system/camerad/cameras/ife.h"

void test_camera_policy() {
  OS04C10 sensor;
  std::array<uint8_t, 8192> program = {};
  std::vector<uint32_t> patches;
  const int program_size = build_initial_config(program.data(), NARROW_ROAD_CAMERA_CONFIG, &sensor, patches, 1344, 760);
  std::vector<IspRegWrite> regs;
  std::vector<IspLutWrite> luts;
  CHECK(parse_isp_cdm(program.data(), program_size, regs, luts));
  std::map<uint32_t, uint32_t> values;
  for (const auto &reg : regs) values[reg.offset] = reg.value;
  CHECK(values.at(0x6b0) == 0x040000c0);  // RAW12 black level and unity scale
  CHECK(values.at(0x6fc) == 0x00800080);  // sensor performs white balance
  CHECK(values.at(0x700) == 0x80);
  CHECK(values.at(0x40) == 0xd06);        // road lens shading enabled
  CHECK(values.at(0xe14) == 1343);
  CHECK(values.at(0xe10) == 759);
  for (size_t i = 0; i < sensor.color_correct_matrix.size(); ++i) {
    CHECK(values.at(0x760 + i * 4) == sensor.color_correct_matrix[i]);
  }
  CHECK(luts.size() == 6);
  CHECK(luts[0].bank == 9 && luts[0].bytes == 36 * 4);
  CHECK(luts[1].bank == 14 && luts[1].bytes == 221 * 4);
  CHECK(luts[2].bank == 15 && luts[2].bytes == 221 * 4);
  for (int i = 3; i < 6; ++i) {
    CHECK(luts[i].bank == 26 + 2 * (i - 3) && luts[i].bytes == 64 * 4);
  }
  // Truncated commands and unknown opcodes must never reach hardware.
  const auto rejects = [](const uint8_t *data, size_t bytes) {
    std::vector<IspRegWrite> r;
    std::vector<IspLutWrite> l;
    return !parse_isp_cdm(data, bytes, r, l);
  };
  CHECK(rejects(program.data(), program_size - 1));
  const std::array<uint8_t, 4> unknown = {0, 0, 0, 0xff};
  CHECK(rejects(unknown.data(), unknown.size()));
  CHECK(rejects(unknown.data(), 3));

  // The same metering regions fit within both sensors' published images.
  for (auto image_size : {std::pair{1344, 760}, std::pair{1928, 1208}}) {
    for (int camera = 0; camera < 3; ++camera) {
      auto r = get_exposure_rect(camera, camera == 1 ? 2000.0 : 427.5, image_size.first, image_size.second);
      CHECK(r.x >= 0 && r.y >= 0 && r.w > 0 && r.h > 0);
      CHECK(r.x + r.w <= image_size.first && r.y + r.h <= image_size.second);
    }
  }

  AutoExposure ae;
  ae.init(&sensor, 1);
  for (uint32_t frame = 1; frame <= 1000; ++frame) {
    ae.update(1.0f / 256, frame);
    CHECK(ae.exposure_time >= sensor.exposure_time_min && ae.exposure_time <= sensor.exposure_time_max);
    CHECK(ae.gain_idx >= sensor.analog_gain_min_idx && ae.gain_idx <= sensor.analog_gain_max_idx);
    CHECK(std::isfinite(ae.cur_ev[frame % 3]));
  }
  CHECK(ae.gain_idx == sensor.analog_gain_max_idx);
  CHECK(ae.exposure_time == sensor.exposure_time_max);
  CHECK(sensor.sensor_analog_gains[ae.gain_idx] == 8.5f);
  for (uint32_t frame = 1001; frame <= 2000; ++frame) ae.update(1.0f, frame);
  CHECK(ae.gain_idx == sensor.analog_gain_min_idx);
  CHECK(ae.exposure_time == sensor.exposure_time_min);
  ae.update(0.2f, 2001, 8, 100);
  CHECK(ae.gain_idx == 8 && ae.exposure_time == 100 && !ae.dc_gain_enabled);

  // A dim scene must reach the brighter target during startup, without the
  // normal ten-second target filter holding it near the initial dark value.
  // Model a three-frame sensor delay in both OS04 exposure-time units.
  for (bool raw10 : {false, true}) {
    OS04C10 startup_sensor;
    if (raw10) {
      startup_sensor.ev_scale = 75;
      startup_sensor.exposure_time_max = 4717;
      startup_sensor.max_ev = 4717 * 8.5f;
    }
    for (int camera : {0, 1}) {
      AutoExposure startup;
      startup.init(&startup_sensor, camera, raw10 ? 600 : 5);
      std::array<float, 3> history;
      history.fill(startup.exposure_time);
      float grey = 0;
      for (uint32_t frame = 1; frame <= 120; ++frame) {
        grey = std::clamp(0.1f + history[frame % 3] * 0.00002f * startup_sensor.ev_scale / 150, 0.0f, 1.0f);
        startup.update(grey, frame);
        history[frame % 3] = startup.exposure_time * startup_sensor.sensor_analog_gains[startup.gain_idx];
        CHECK(startup.exposure_time <= startup_sensor.exposure_time_max);
        CHECK(startup.gain_idx <= startup_sensor.analog_gain_max_idx);
        if (frame >= 60) CHECK(std::abs(grey - 0.15f) < 0.01f);
      }
      // A subsequent bright scene must reduce exposure within the same limits.
      const float dim_ev = history[0];
      for (uint32_t frame = 121; frame <= 240; ++frame) {
        grey = std::clamp(0.1f + history[frame % 3] * 0.002f * startup_sensor.ev_scale / 150, 0.0f, 1.0f);
        startup.update(grey, frame);
        history[frame % 3] = startup.exposure_time * startup_sensor.sensor_analog_gains[startup.gain_idx];
        CHECK(startup.exposure_time >= startup_sensor.exposure_time_min);
        CHECK(startup.gain_idx >= startup_sensor.analog_gain_min_idx);
      }
      CHECK(history[0] < dim_ev / 10);
      CHECK(grey < 0.5f);
    }
  }
}

int main() { return run_native_test(test_camera_policy); }
