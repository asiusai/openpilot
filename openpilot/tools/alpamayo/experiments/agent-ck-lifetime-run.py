"""Fresh entry for the fixed helper ownership, using the frozen original runner."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('ck_lifetime_overlay',Path(__file__).with_name('agent-ck-lifetime-overlay.py'))
overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
if __name__=='__main__': overlay.load_runner()['main']()
