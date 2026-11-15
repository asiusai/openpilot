"""Package exact upstream sources and the already downloaded evaluation assets."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
from PIL import Image

base = Path(os.environ['ALPAMAYO_ASSET_ROOT'])
dest = base / 'car-bundle'
dest.mkdir(exist_ok=True)
revisions = {
  'openpilot': 'fe68e1d0595d48ffd0d058404181d873250e4ae4',
  'tinygrad': '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e',
  'upstream': '02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e',
  'model': 'b92b6aa326bd3e2235db28f248ad5a30b5890e64',
}
for name in ('openpilot', 'tinygrad', 'upstream'):
  args = ['git', '-C', str(base / name), 'archive', revisions[name]]
  if name == 'openpilot':
    args.append('openpilot/tools/alpamayo')
  archive = subprocess.check_output(args)
  target = dest / name
  target.mkdir(exist_ok=True)
  with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(target, filter='data')
if not (dest / 'alpamayo').exists():
  (dest / 'openpilot/openpilot/tools/alpamayo').rename(dest / 'alpamayo')
else:
  shutil.copytree(dest / 'openpilot/openpilot/tools/alpamayo', dest / 'alpamayo', dirs_exist_ok=True)

(dest / 'model').mkdir(exist_ok=True)
for source in (base / 'model').iterdir():
  if source.is_file():
    target = dest / 'model' / source.name
    if not target.exists():
      os.link(source, target)

(dest / 'samples').mkdir(exist_ok=True)
(dest / 'references').mkdir(exist_ok=True)
for seconds in (5, 25, 45):
  with np.load(base / 'crops' / f'sample-{seconds}.npz', allow_pickle=False) as sample:
    original = {k: sample[k] for k in sample.files}
  for variant in ('original', 'narrow30'):
    frames = original['image_frames'].copy()
    if variant == 'narrow30':
      h, w = frames.shape[-2:]
      fraction = 2 * 1141.5 * np.tan(np.deg2rad(15)) / 1344
      cw, ch = round(w * fraction), round(h * fraction)
      left, top = (w-cw)//2, (h-ch)//2
      for i in range(5, 10):
        image = Image.fromarray(frames[i, 0].transpose(1, 2, 0))
        image = image.crop((left, top, left+cw, top+ch)).resize((w, h), Image.Resampling.LANCZOS)
        frames[i, 0] = np.asarray(image).transpose(2, 0, 1)
    np.savez_compressed(dest / 'samples' / f'{seconds}-{variant}.npz', **(original | {'image_frames': frames}))
    baseline = 'original' if variant == 'original' else 'narrow30_nominal'
    shutil.copy2(base / 'crops' / f'result-{seconds}-{baseline}.json', dest / 'references' / f'{seconds}-{variant}.json')
  shutil.copy2(base / 'crops' / f'reference-{seconds}.json', dest / 'references' / f'motion-{seconds}.json')

(dest / 'revisions.json').write_text(json.dumps(revisions, indent=2) + '\n')
for name in ('preflight.py', 'run-parked.sh', 'summarize.py', 'prepare-fp8.py', 'benchmark-fast.py',
             'shadow-capture.py', 'shadow-worker.py', 'run-shadow.py', 'README.md'):
  shutil.copy2(Path(__file__).parent / name, dest / name)
checksums = []
for path in sorted(dest.rglob('*')):
  if path.is_file() and path.name != 'SHA256SUMS':
    with path.open('rb') as f:
      digest = hashlib.file_digest(f, 'sha256').hexdigest()
    checksums.append(f'{digest}  {path.relative_to(dest)}')
(dest / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
print(dest)
