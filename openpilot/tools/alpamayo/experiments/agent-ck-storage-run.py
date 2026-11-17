"""Fresh entry for the exact frozen first-layer runner with the storage-field correction."""
import importlib.util
from pathlib import Path

path=Path(__file__).with_name('agent-ck-storage-overlay.py')
spec=importlib.util.spec_from_file_location('ck_storage_overlay',path)
overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)

if __name__=='__main__': overlay.load_runner()['main']()
