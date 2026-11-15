# Synthetic recording fixtures

The MP4 files contain three seconds of generated 64x64 YUV420 video at 20 fps.
The raw H.264 and HEVC files contain the first second of the corresponding MP4 stream.
Luma alternates each second; chroma remains neutral. Each second starts with an IDR frame. There are no B-frames or captured recordings.

Tests remux MP4 files with stream copy and read raw frames using the committed packet index.
The index binds packet boundaries to each raw file's SHA-256 hash.
Tests need FFmpeg's MP4 demuxer, muxer, and decoders, but no software encoder or raw input/output format support.
This keeps H.264 and HEVC recording, encryption, and upload tests available on devices with hardware encoding.

Regenerate on a host with FFmpeg, libx264, and libx265:

```sh
python openpilot/system/loggerd/tests/fixtures/generate.py
```

The committed files were generated with Ubuntu FFmpeg 6.1.1-3ubuntu5. The generator prints their SHA-256 hashes.
Encoder versions can change compressed bytes. Review regenerated streams before replacing the fixtures.
