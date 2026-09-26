import sys
import time
from pathlib import Path

from tinygrad import Tensor, Device

from openpilot.common.hardware import AGNOS, ASIUS_HARDWARE
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def tensor_from_dma_buf(ptr: int, fd: int | None, size: int, device: str) -> Tensor:
  if fd is None or device.split(':', 1)[0] != 'QCOM':
    return Tensor.from_blob(ptr, (size,), dtype='uint8', device=device)
  tensor = Tensor.empty(size, dtype='uint8', device=device)
  buffer = tensor.uop.buffer
  buffer.allocate(Device[device].iface.map(ptr, buffer.nbytes, fd))
  return tensor


def modeld_pkl_path(chestnut: bool):
  if chestnut and ASIUS_HARDWARE:
    return MODELS_DIR / 'big_driving_asius_tinygrad.pkl'
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def load_oob(path, chestnut=False):
  from tinygrad import Context
  device = 'USB+AMD:LLVM' if chestnut else 'QCOM' if AGNOS or ASIUS_HARDWARE else 'METAL' if sys.platform == 'darwin' else 'CPU:LLVM'
  with Context(DEV=device):
    from tinygrad_repo.examples.openpilot.helpers import load_pickle
    return load_pickle(path, out_of_band=True)

def chestnut_present(timeout: float = 0.) -> bool:
  deadline = time.monotonic() + timeout
  seen = False
  while True:
    for d in USB_DEVICES_PATH.glob("*"):
      try:
        usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
        product = (d / "product").read_text().strip()
        if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
          seen = True
          if float((d / "speed").read_text()) >= 5000:
            return True
      except (OSError, ValueError):
        pass
    if not seen or time.monotonic() >= deadline:
      return False
    # Keep USB idle during firmware recovery, including its brief disconnect.
    # F3/GPU initialization at 480M would cancel the pending SuperSpeed retry.
    time.sleep(.1)

def chestnut_compiled() -> bool:
  return modeld_pkl_path(chestnut=True).is_file() and all(
    (MODELS_DIR / f'big_driving_warp_{size}_tinygrad.pkl').is_file() for size in ('1344x760', '1928x1208'))
