"""Explicit installation into a new directory, without changing openpilot dependencies."""
import argparse
import os
from pathlib import Path
import subprocess
import tomllib

from openpilot.selfdrive.alpamayo.inference import NVIDIA_REVISION, TINYGRAD_REVISION


def install_dependencies(project):
  packages = {package['name']: package for package in tomllib.loads((project/'uv.lock').read_text())['package']}
  antlr = packages['antlr4-python3-runtime']
  # This reviewed source distribution uses ordinary setuptools without custom build hooks.
  if antlr['version'] != '4.9.3' or antlr['sdist']['hash'] != 'sha256:f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b':
    raise RuntimeError('Review the changed ANTLR build before installation')
  environment = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(project/'.venv'), UV_PYTHON_DOWNLOADS='never')
  subprocess.run(['uv', 'venv', '--python', '3.12', str(project/'.venv')], env=environment, check=True)
  python = project/'.venv/bin/python'
  setuptools = packages['setuptools']
  requirement = f'setuptools=={setuptools["version"]} '+ ' '.join(f'--hash={wheel["hash"]}' for wheel in setuptools['wheels'])+'\n'
  subprocess.run(['uv', 'pip', 'install', '--python', str(python), '--no-deps', '--only-binary=:all:', '--require-hashes', '-r', '-'],
                 input=requirement, text=True, env=environment, check=True)
  command = ['uv', 'sync', '--frozen', '--no-dev', '--no-install-project', '--no-build-isolation-package', antlr['name'],
             '--project', str(project)]
  for name in sorted(packages.keys()-{antlr['name']}):
    command.extend(['--no-build-package', name])
  subprocess.run(command, env=environment, check=True)
  subprocess.run(['uv', 'pip', 'install', '--python', str(python), '--no-deps', '--only-binary=:all:', '--require-hashes',
                  '-r', str(Path(__file__).with_name('extras.txt'))], env=environment, check=True)
  site = Path(subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
  # Import the pinned checkout directly, without invoking its unpinned hatchling build environment.
  (site/'alpamayo-source.pth').write_text(str(project/'src')+'\n')
  return python


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('directory', type=Path)
  args = parser.parse_args()
  root = args.directory.resolve()
  root.mkdir(parents=True, exist_ok=False)
  for name, url, revision in (
      ('nvidia', 'https://github.com/NVlabs/alpamayo2-edge.git', NVIDIA_REVISION),
      ('tinygrad', 'https://github.com/commaai/tinygrad.git', TINYGRAD_REVISION)):
    path = root/name
    subprocess.run(['git', 'init', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', url], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth=1', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
  python = install_dependencies(root/'nvidia')
  print(f'Python: {python}\nIsolated tinygrad: {root / "tinygrad"}')


if __name__ == '__main__':
  main()
