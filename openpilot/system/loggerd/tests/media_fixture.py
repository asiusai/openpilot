import subprocess
from pathlib import Path


def make_video(path, seconds, codec='libx264', format_name=None):
  # Remux committed synthetic frames; device tests need no software encoder.
  if seconds not in (1, 2, 3):
    raise ValueError('Video fixtures contain three seconds at 20 fps')
  fixture = Path(__file__).parent / 'fixtures' / {'libx264': 'h264.mp4', 'libx265': 'hevc.mp4'}[codec]
  command = ['ffmpeg', '-v', 'error', '-i', str(fixture), '-map', '0:v:0', '-c:v', 'copy', '-frames:v', str(seconds * 20)]
  if format_name:
    command += ['-f', format_name]
  else:
    command += ['-movflags', '+frag_keyframe+empty_moov+default_base_moof']
  subprocess.run([*command, str(path)], check=True, stderr=subprocess.PIPE)


def decrypt_recording(path, key):
  """Decode encrypted samples for FFmpeg, whose CENC demuxer only accepts AES-128.

  The test still decodes every restored frame and compares its pixel hash.
  Browser integration separately uses the production TypeScript MP4 decoder.
  """
  import struct
  from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  from openpilot.system.loggerd.data_media import read_boxes, child, parse_tfhd, parse_trun, fullbox_flags

  data = bytearray(path.read_bytes())
  for moof in (b for b in read_boxes(data, 0, len(data)) if b.type == b'moof'):
    for traf in (b for b in read_boxes(data, moof.payload_start, moof.end) if b.type == b'traf'):
      _, base, duration, size = parse_tfhd(data, child(data, traf, b'tfhd'), moof.start)
      senc = child(data, traf, b'senc')
      aux = senc.payload_start + 8
      count = 0
      previous = None
      for trun in (b for b in read_boxes(data, traf.payload_start, traf.end) if b.type == b'trun'):
        run = parse_trun(data, trun, duration, size, previous, base)
        cursor = run.data_offset
        for sample_size in run.sample_sizes:
          start = cursor
          cipher = Cipher(algorithms.AES(key), modes.CTR(bytes(data[aux:aux + 16]))).decryptor()
          aux += 16
          if fullbox_flags(data, senc) & 2:
            entries = struct.unpack_from('>H', data, aux)[0]
            aux += 2
            for _ in range(entries):
              clear, encrypted = struct.unpack_from('>HI', data, aux)
              aux += 6
              cursor += clear
              data[cursor:cursor + encrypted] = cipher.update(bytes(data[cursor:cursor + encrypted]))
              cursor += encrypted
          else:
            data[cursor:cursor + sample_size] = cipher.update(bytes(data[cursor:cursor + sample_size]))
            cursor += sample_size
          cipher.finalize()
          assert cursor == start + sample_size
          count += 1
        previous = cursor
      assert aux == senc.end and count == struct.unpack_from('>I', data, senc.payload_start + 4)[0]
  decoded = path.with_suffix('.clear.mp4')
  decoded.write_bytes(data)
  return decoded
