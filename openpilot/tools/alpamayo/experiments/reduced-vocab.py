"""Optional NVIDIA CoC vocabulary; equivalence requires the full-head argmax in this set.

The input embedding remains complete. Original weight quantization and scales remain unchanged.
"""
import hashlib
import json
from pathlib import Path
from types import MethodType
from tinygrad import Tensor, dtypes


def install(model, path):
  raw = Path(path).read_bytes()
  vocab = sorted(set(json.loads(raw)))
  config = model.config
  def allowed(i):
    return (not config['traj_ids']['history_id0'] <= i < config['traj_ids']['history_id0']+config['history_vocab_size']
            and not config['traj_ids']['future_id0'] <= i < config['traj_ids']['future_id0']+config['future_vocab_size']
            and i != config['vlm_config']['text_config']['eos_token_id'])
  vocab = [i for i in vocab if allowed(i)]
  assert config['traj_ids']['future_start'] in vocab
  assert vocab and min(vocab) >= 0 and max(vocab) < model.weights['vlm.lm_head.weight'].shape[0]
  # Repeating the final row pads GEMV without introducing a new output token.
  padded = vocab + [vocab[-1]]*((-len(vocab)) % 64)
  mapping = Tensor(padded, dtype=dtypes.int32).realize()
  model.output_token_mapping = mapping
  model.weights['vlm.lm_head.weight'] = model.weights['vlm.lm_head.weight'][mapping].contiguous().realize()
  def greedy(self, logits, cos, sin, position, step):
    token = mapping[logits.reshape(-1).argmax()].realize()
    x = self.weights['vlm.model.language_model.embed_tokens.weight'][token].reshape(1, 1, -1)
    logits = self._language(x, cos[:, :, step:step+1], sin[:, :, step:step+1], position)
    return logits, token.float().realize()
  model._greedy_decode = MethodType(greedy, model)
  from tinygrad import TinyJit
  model.greedy_decode_jit = TinyJit(model._greedy_decode)
  return {'sha256': hashlib.sha256(raw).hexdigest(), 'tokens': len(vocab), 'padded_rows': len(padded),
          'limitation': 'Matches full-head greedy decoding only when its argmax belongs to this vocabulary.'}
