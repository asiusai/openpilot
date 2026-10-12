"""Catch runtime/model mismatches before installing an upstream rebase."""
import pickle

import numpy as np
import pytest
from tinygrad import Context, Tensor, dtypes

from openpilot.selfdrive.modeld.helpers import MODELS_DIR
from openpilot.selfdrive.modeld.modeld import input_view


@pytest.mark.parametrize('dtype,np_dtype,offset', [(dtypes.float32, np.float32, 16), (dtypes.uint8, np.uint8, 4)])
def test_model_input_view_shares_typed_buffer(dtype, np_dtype, offset):
  parent = Tensor(np.arange(16, dtype=np_dtype), device='CPU').realize()
  view = input_view(parent._buffer(), (2, 2), dtype, offset)
  assert view.dtype == dtype
  np.testing.assert_array_equal(view.numpy(), [[4, 5], [6, 7]])
  replacement = Tensor(np.arange(16, dtype=np_dtype) + 10, device='CPU').realize()
  parent._buffer().copy_from(replacement._buffer())
  np.testing.assert_array_equal(view.numpy(), [[14, 15], [16, 17]])


@pytest.mark.parametrize('name', ['big_driving_tinygrad.pkl', 'big_driving_warp_1344x760_tinygrad.pkl', 'big_driving_warp_1928x1208_tinygrad.pkl'])
def test_chestnut_artifact_matches_runtime(name):
  path = MODELS_DIR / name
  if not path.exists() or path.stat().st_size < 1024:
    pytest.skip('Fetch the LFS model artifacts first')
  # Deserialize the real artifacts without allocating or running a GPU.
  # Execution and timing still require hardware validation.
  with Context(DEV='NULL'):
    if name == 'big_driving_tinygrad.pkl':
      from tinygrad_repo.examples.openpilot.helpers import load_pickle
      artifact = load_pickle(path, out_of_band=True)
    else:
      with path.open('rb') as source:
        artifact = pickle.load(source)
    assert artifact['input_specs']
    assert artifact['run'].captured is not None
