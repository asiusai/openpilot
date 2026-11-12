"""Regenerate synthetic recording fixtures on a host with libx264 and libx265."""

import hashlib
from pathlib import Path
import subprocess


def main():
  frames = b''.join(bytes([16 + index // 20 % 2 * 64]) * (64 * 64) + bytes([128]) * (64 * 64 // 2) for index in range(60))
  for name, codec in (('h264', 'libx264'), ('hevc', 'libx265')):
    path = Path(__file__).parent / f'{name}.mp4'
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '64x64', '-r', '20', '-i', 'pipe:0',
               '-c:v', codec, '-threads', '1', '-g', '20', '-bf', '0']
    if codec == 'libx265':
      command += ['-x265-params', 'pools=none:frame-threads=1:scenecut=0:open-gop=0:log-level=error']
    else:
      command += ['-sc_threshold', '0']
    command += ['-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(path)]
    subprocess.run(command, input=frames, check=True)
    print(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}')
    raw = path.with_suffix(f'.{name}')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-c:v', 'copy',
                    '-frames:v', '20', '-f', name, str(raw)], check=True)
    print(f'{hashlib.sha256(raw.read_bytes()).hexdigest()}  {raw.name}')


if __name__ == '__main__':
  main()
