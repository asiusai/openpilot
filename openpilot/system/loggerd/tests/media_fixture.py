import subprocess

import pytest


def make_video(path, seconds, codec='libx264', format_name=None):
  # The device's small FFmpeg build has rawvideo, but no lavfi test sources.
  if codec not in subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.DEVNULL, text=True):
    pytest.skip(f'Fixture encoder {codec} is unavailable')
  frames = b''.join(bytes([16 + index // 20 % 2 * 64]) * (64 * 64) + bytes([128]) * (64 * 64 // 2) for index in range(seconds * 20))
  command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '64x64', '-r', '20', '-i', 'pipe:0',
             '-c:v', codec, '-threads', '1', '-g', '20', '-bf', '0']
  if format_name:
    command += ['-f', format_name]
  else:
    command += ['-movflags', '+frag_keyframe+empty_moov+default_base_moof']
  subprocess.run([*command, str(path)], input=frames, check=True, stderr=subprocess.PIPE)
