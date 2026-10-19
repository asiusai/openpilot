#include <cmath>
#include <filesystem>
#include <vector>

#include "common/tests/native_test.h"
#include "system/loggerd/video_writer.h"

// Exercise the production muxer with a late keyframe, capture gaps, and audio
// arriving before video. Input is a generated H.264 fixture, never a driving log.
int main(int argc, char **argv) {
  return run_native_test([&] {
    REQUIRE(argc == 4);
    AVFormatContext *input = nullptr;
    REQUIRE(avformat_open_input(&input, argv[1], nullptr, nullptr) == 0);
    REQUIRE(avformat_find_stream_info(input, nullptr) >= 0);
    const int track = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    REQUIRE(track >= 0);
    const auto codec = input->streams[track]->codecpar;
    const std::filesystem::path output(argv[2]);
    constexpr int sample_rate = 48000;
    constexpr int64_t origin = 3600000000LL;
    {
      VideoWriter writer(output.parent_path().c_str(), output.filename().c_str(), true,
                         codec->width, codec->height, 20, cereal::EncodeIndex::Type::QCAMERA_H264);
      writer.set_metadata("ai.asius.startOffsetMs", "46000");
      int64_t audio_time = origin + std::stoll(argv[3]);
      auto audio = [&] {
        std::vector<int16_t> samples(sample_rate / 20);
        for (size_t i = 0; i < samples.size(); ++i) samples[i] = std::sin(2 * M_PI * 440 * i / sample_rate) * 8000;
        writer.write_audio(reinterpret_cast<uint8_t *>(samples.data()), samples.size() * sizeof(int16_t), audio_time, sample_rate);
        audio_time += 50000;
      };
      audio();
      audio();
      writer.write(codec->extradata, codec->extradata_size, origin, true, false);
      AVPacket *packet = av_packet_alloc();
      int frame = 0;
      while (av_read_frame(input, packet) >= 0) {
        if (packet->stream_index == track) {
          // Retain a 200 ms capture gap after frame 19.
          const int64_t timestamp = origin + frame * 50000LL + (frame >= 20 ? 200000 : 0);
          while (audio_time < timestamp + 50000) audio();
          writer.write(packet->data, packet->size, timestamp, false, packet->flags & AV_PKT_FLAG_KEY);
          ++frame;
        }
        av_packet_unref(packet);
      }
      // Trailing audio must not extend this video by several seconds.
      for (int i = 0; i < 20; ++i) audio();
      av_packet_free(&packet);
    }
    avformat_close_input(&input);
  });
}
