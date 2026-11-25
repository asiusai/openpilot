#!/usr/bin/env python3
"""Install what the Alpamayo planner needs under /data/alpamayo and write its configuration.

The manager runs this offroad until `complete()` is true. Steps:
1. The isolated runtime: pinned tinygrad and NVIDIA Alpamayo 2 Edge sources with their locked Python dependencies.
2. The model files of `release.json` from Hugging Face, each checked against its pinned SHA-256.
3. The pinned AMD GPU firmware and the compiled GPU graphs for this code and model.
4. `config.json`, the configuration that the planner reads.

Every step keeps verified work, so an interrupted install continues. The normal openpilot environment stays unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request

from openpilot.selfdrive.alpamayo.inference import NVIDIA_REVISION, TINYGRAD_REVISION

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('ALPAMAYO_ROOT', '/data/alpamayo'))
CONFIG = ROOT/'config.json'
STATUS = ROOT/'status.json'
RELEASE = HERE/'release.json'
SOURCES = (('nvidia', 'https://github.com/NVlabs/alpamayo2-edge.git', NVIDIA_REVISION),
           ('tinygrad', 'https://github.com/commaai/tinygrad.git', TINYGRAD_REVISION))
RETRY_S = 60


OVERRIDES = ROOT/'overrides.json'
_PINNED = None


def identity():
  """Changes when the pinned model, runtime sources, dependencies or the user's overrides change."""
  global _PINNED
  if _PINNED is None:
    _PINNED = [hashlib.sha256(part).digest() for part in
               (RELEASE.read_bytes(), (HERE/'extras.txt').read_bytes(), NVIDIA_REVISION.encode(), TINYGRAD_REVISION.encode())]
  digest = hashlib.sha256()
  for part in (*_PINNED, hashlib.sha256(OVERRIDES.read_bytes() if OVERRIDES.is_file() else b'').digest()):
    digest.update(part)
  return digest.hexdigest()


def config_path():
  """The installed configuration, or the file that ALPAMAYO_CONFIG names."""
  return Path(os.environ.get('ALPAMAYO_CONFIG') or CONFIG)


def managed():
  """True when the planner reads the configuration that this installer writes."""
  return config_path() == CONFIG


def complete():
  try:
    return CONFIG.is_file() and (ROOT/'installed').read_text().strip() == identity()
  except OSError:
    return False


def ready():
  """The configured planner can start: its configuration exists and, when this installer owns it, matches this code."""
  return complete() if managed() else config_path().is_file()


def write(path, text):
  temporary = path.with_name(path.name+'.tmp')
  temporary.write_text(text)
  os.replace(temporary, path)


def status(step, text):
  """Progress for the log and for the app."""
  print(f'alpamayo install: {step}: {text}', flush=True)
  write(STATUS, json.dumps({'step': step, 'text': text, 'time': time.time()})+'\n')


def file_sha256(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(path, sha256, size):
  """Hash a file once; later checks compare its size and modification time with the record."""
  if not path.is_file() or path.stat().st_size != size:
    return False
  records_path = ROOT/'verified.json'
  records = json.loads(records_path.read_text()) if records_path.is_file() else {}
  stat = path.stat()
  record = [sha256, stat.st_size, stat.st_mtime_ns]
  if records.get(str(path)) == record:
    return True
  if file_sha256(path) != sha256:
    return False
  records[str(path)] = record
  write(records_path, json.dumps(records, indent=1)+'\n')
  return True


def download(url, path, sha256, size, label):
  """Download with resume into `path`. The file appears only after its SHA-256 matches."""
  if verified(path, sha256, size):
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  part = path.with_name(path.name+'.part')
  if part.is_file() and part.stat().st_size > size:
    part.unlink()
  digest, done = hashlib.sha256(), 0
  if part.is_file():
    with part.open('rb') as stream:
      while chunk := stream.read(1 << 20):
        digest.update(chunk)
        done += len(chunk)
  if done < size:
    request = urllib.request.Request(url, headers={'Range': f'bytes={done}-', 'User-Agent': 'openpilot-alpamayo-install'})
    with urllib.request.urlopen(request, timeout=60) as response:
      if done and response.status != 206:  # the server ignored the range: start again
        digest, done = hashlib.sha256(), 0
      shown = time.monotonic()
      with part.open('ab' if done else 'wb') as out:
        while chunk := response.read(1 << 20):
          out.write(chunk)
          digest.update(chunk)
          done += len(chunk)
          if time.monotonic()-shown > 5:
            shown = time.monotonic()
            status('download', f'{label} {100*done//size}% of {size/1e6:.0f} MB')
  if done != size or digest.hexdigest() != sha256:
    part.unlink(missing_ok=True)
    raise RuntimeError(f'{label} differs from its pinned SHA-256')
  os.replace(part, path)
  verified(path, sha256, size)


def install_dependencies(project, environment):
  packages = {package['name']: package for package in tomllib.loads((project/'uv.lock').read_text())['package']}
  antlr = packages['antlr4-python3-runtime']
  # This reviewed source distribution uses ordinary setuptools without custom build hooks.
  if antlr['version'] != '4.9.3' or antlr['sdist']['hash'] != 'sha256:f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b':
    raise RuntimeError('Review the changed ANTLR build before installation')
  environment = dict(environment, UV_PROJECT_ENVIRONMENT=str(project/'.venv'), UV_PYTHON_DOWNLOADS='never')
  uv = shutil.which('uv', path=environment['PATH'])
  # The environment uses openpilot's own interpreter version, not another Python 3.12 that uv finds first.
  if sys.version_info[:2] != (3, 12):
    raise RuntimeError('The Alpamayo runtime requires Python 3.12')
  subprocess.run([uv, 'venv', '--python', sys.executable, str(project/'.venv')], env=environment, check=True)
  python = project/'.venv/bin/python'
  setuptools = packages['setuptools']
  requirement = f'setuptools=={setuptools["version"]} '+' '.join(f'--hash={wheel["hash"]}' for wheel in setuptools['wheels'])+'\n'
  subprocess.run([uv, 'pip', 'install', '--python', str(python), '--no-deps', '--only-binary=:all:', '--require-hashes', '-r', '-'],
                 input=requirement, text=True, env=environment, check=True)
  command = [uv, 'sync', '--frozen', '--no-dev', '--no-install-project', '--no-build-isolation-package', antlr['name'],
             '--project', str(project)]
  for name in sorted(packages.keys()-{antlr['name']}):
    command.extend(['--no-build-package', name])
  subprocess.run(command, env=environment, check=True)
  subprocess.run([uv, 'pip', 'install', '--python', str(python), '--no-deps', '--only-binary=:all:', '--require-hashes',
                  '-r', str(HERE/'extras.txt')], env=environment, check=True)
  site = Path(subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
  # Import the pinned checkout directly, without invoking its unpinned hatchling build environment.
  (site/'alpamayo-source.pth').write_text(str(project/'src')+'\n')


def install_runtime(root):
  """Pinned sources and a frozen, hash-checked Python environment. Existing Python 3.12 only; nothing is built except ANTLR."""
  marker = root/'installed'
  wanted = f'{NVIDIA_REVISION} {TINYGRAD_REVISION} {hashlib.sha256((HERE/"extras.txt").read_bytes()).hexdigest()}'
  if marker.is_file() and marker.read_text().strip() == wanted:
    return
  status('runtime', 'fetching the pinned tinygrad and NVIDIA sources')
  shutil.rmtree(root, ignore_errors=True)
  root.mkdir(parents=True)
  for name, url, revision in SOURCES:
    path = root/name
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', url], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', '-q', '--depth=1', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '-q', '--detach', revision], check=True)
  status('runtime', 'installing the locked Python packages (about 1 GB)')
  uv_cache = root/'uv-cache'
  # The device puts wrapper scripts for system-wide installs before the real tools: /usr/comma/shims/uv remounts the root
  # filesystem and runs a system uv. Use the real uv, and keep openpilot's own environment settings away from it.
  environment = {k: v for k, v in os.environ.items() if k not in ('VIRTUAL_ENV', 'PYTHONPATH') and not k.startswith('UV_')}
  environment['PATH'] = os.pathsep.join(d for d in os.environ.get('PATH', os.defpath).split(os.pathsep) if not d.endswith('/shims'))
  if shutil.which('uv', path=environment['PATH']) is None:
    raise RuntimeError('uv is not installed')
  install_dependencies(root/'nvidia', dict(environment, UV_CACHE_DIR=str(uv_cache), UV_LINK_MODE='copy'))
  shutil.rmtree(uv_cache, ignore_errors=True)
  write(marker, wanted+'\n')


def install_model(release, model):
  base = f'https://huggingface.co/{release["repository"]}/resolve/{release["revision"]}/{release["prefix"]}'
  for name, pin in release['files'].items():
    download(base+name, model/name, pin['sha256'], pin['bytes'], name)
  noise = model/'initial_noise-zero.npy'  # the model is trained and validated with zero initial diffusion noise
  if not noise.is_file():
    import numpy as np
    np.save(noise, np.zeros((1, 64, 2), np.float32))


def install_firmware(release, cache):
  firmware = release['firmware']
  for name, sha256 in firmware['files'].items():
    url = firmware['url']+name
    path = cache/'tinygrad/downloads/fw'/hashlib.md5(url.encode()).hexdigest()  # tinygrad's download cache name
    if not path.is_file() or file_sha256(path) != sha256:
      path.parent.mkdir(parents=True, exist_ok=True)
      status('firmware', name)
      with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'openpilot-alpamayo-install'}), timeout=60) as response:
        raw = response.read()
      if hashlib.sha256(raw).hexdigest() != sha256:
        raise RuntimeError(f'GPU firmware {name} differs from its pinned SHA-256')
      temporary = path.with_name(path.name+'.tmp')
      temporary.write_bytes(raw)
      os.replace(temporary, path)


def install_graphs(release, cache, target='gfx1200'):
  """Compiled GPU graphs and kernels for this code, model and GPU type. Without them the first start compiles for 10-20 min."""
  pin = release['graph_cache'].get(target)
  marker = cache/'graph-cache.installed'
  if pin is None or (marker.is_file() and marker.read_text().strip() == pin['sha256']):
    return
  archive = ROOT/'downloads'/Path(pin['path']).name
  base = f'https://huggingface.co/{release["repository"]}/resolve/{release["revision"]}/{release["prefix"]}'
  download(base+pin['path'], archive, pin['sha256'], pin['bytes'], 'compiled GPU graphs')
  status('graphs', 'unpacking the compiled GPU graphs')
  staging = cache/'graph-cache.tmp'
  shutil.rmtree(staging, ignore_errors=True)
  with tarfile.open(archive) as tar:
    tar.extractall(staging, filter='data')
  shutil.rmtree(cache/'graphs', ignore_errors=True)
  for stale in cache.glob('tinygrad.db*'):
    stale.unlink()
  for item in staging.iterdir():
    os.replace(item, cache/item.name)
  staging.rmdir()
  write(marker, pin['sha256']+'\n')
  archive.unlink()


def merge(base, override):
  for key, value in override.items():
    base[key] = merge(dict(base.get(key, {})), value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
  return base


def configuration(release):
  """The pinned defaults with this device's paths. `overrides.json` changes single settings."""
  model = ROOT/'models'/release['model']
  config = json.loads(json.dumps(release['config']))
  config.update(python=str(ROOT/'runtime/nvidia/.venv/bin/python'), tinygrad=str(ROOT/'runtime/tinygrad'),
                cache=str(ROOT/'cache'), output=str(ROOT/'output'))
  config['inference'].update(variant='selected', checkpoint=str(model/'checkpoint'), fp8_cache=str(model/'fp8-cache'),
                             prefill_calibration=str(model/'prefill-scales.json'), initial_noise=str(model/'initial_noise-zero.npy'))
  if OVERRIDES.is_file():
    merge(config, json.loads(OVERRIDES.read_text()))
  return config


def install():
  release = json.loads(RELEASE.read_text())
  for name in ('cache', 'output', 'downloads', 'models'):
    (ROOT/name).mkdir(parents=True, exist_ok=True)
  install_runtime(ROOT/'runtime')
  status('download', 'model files')
  install_model(release, ROOT/'models'/release['model'])
  install_firmware(release, ROOT/'cache')
  install_graphs(release, ROOT/'cache')
  for old in (ROOT/'models').iterdir():  # weights of an earlier release
    if old.name != release['model']:
      shutil.rmtree(old, ignore_errors=True)
  write(CONFIG, json.dumps(configuration(release), indent=1)+'\n')
  write(ROOT/'installed', identity()+'\n')
  status('done', f'{release["model"]} installed')


def main():
  """Manager entry: retry until the install completes. Network loss and power cycles only delay it."""
  while not complete():
    try:
      install()
    except Exception as error:
      try:
        status('waiting', f'{type(error).__name__}: {error}; next attempt in {RETRY_S} s')
      except OSError:
        print(f'alpamayo install failed: {error}', file=sys.stderr, flush=True)
      time.sleep(RETRY_S)


if __name__ == '__main__':
  if '--once' in sys.argv[1:]:
    install()
  else:
    main()
