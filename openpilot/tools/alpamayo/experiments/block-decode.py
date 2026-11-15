"""Run exact autoregressive tokens in one GPU submission, then check the anchor.

This is not speculative decoding. Tokens after the first anchor are discarded;
the expert's existing cache-length mask excludes their K/V entries.
The pipeline body follows Harald's pinned benchmark.run_native.
"""
import time
import numpy as np
import torch
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp
from benchmark import timed, UnicycleAccelCurvatureCourseActionSpace
from runtime import download


class BlockDecoder:
  def __init__(self, model, block=6):
    self.model, self.block = model, block
    self.jits = {}
    self.readback = download
    self.direct_readback = False
    self.before_diffusion = lambda: None

  def decode(self, logits, cos, sin, position, step, count):
    key = (cos.shape, count, self.model.cache[0].shape[3])
    if key not in self.jits:
      def run(logits, c, s, pos, offset):
        tokens = []
        for i in range(count):
          logits, token = self.model._greedy_decode(logits, c, s, pos+i, offset+i)
          tokens.append(token.reshape(1))
        return logits, Tensor.cat(*tokens).realize()
      self.jits[key] = TinyJit(run)
    position = UOp.variable('block_position', 0, self.model.cache[0].shape[3]-count).bind(position)
    step = UOp.variable('block_step', 0, cos.shape[2]-count).bind(step)
    return self.jits[key](logits, cos, sin, position, step)

  def run(self, model, args, config, tokenizer, prepared, packed, seed, inputs):
    assert args.greedy
    times, generated = {}, []
    rng = np.random.default_rng(seed)
    noise = getattr(args, 'initial_noise', None)
    if noise is None:
      noise = rng.standard_normal((1,64,2))
    started = time.perf_counter()
    with timed('input_upload', times, model.synchronize):
      pixels, vpos, ids, cos, sin, dc, ds, ac, ass, actions = inputs.upload(noise)
    with timed('vision', times, model.synchronize):
      embedding = model.encode(pixels, vpos, ids, inputs.batches, inputs.spans)
    with timed('prefill', times, model.synchronize):
      logits = model.prefill(embedding, cos, sin, args.max_tokens)
    with timed('reasoning', times, model.synchronize):
      generated = self.generate(model, args, config, tokenizer, logits, dc, ds, ids.shape[1])
    self.before_diffusion()
    with timed('diffusion', times, model.synchronize):
      actions = model.trajectory(actions, ac, ass, ids.shape[1]+len(generated), args.diffusion_steps, len(generated))
      action_array = self.readback(actions)
    with timed('trajectory_decode', times, model.synchronize):
      action_config = dict(config.expert_config.action_space_cfg)
      action_config.pop('_target_')
      action_space = UnicycleAccelCurvatureCourseActionSpace(**action_config)
      with torch.no_grad():
        xyz, rot = action_space.action_to_traj(torch.from_numpy(action_array.copy()), prepared['ego_history_xyz'][:,0], prepared['ego_history_rot'][:,0])
      assert xyz.shape == (1,64,3) and rot.shape == (1,64,3,3)
      assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()
    times['total'] = (time.perf_counter()-started)*1000
    return {'timings_ms': times, 'generated_tokens': len(generated), 'generated_token_ids': generated,
            'reasoning': tokenizer.decode(generated),
            'xyz': xyz.numpy().tolist(), 'rot': rot.numpy().tolist()}

  def generate(self, model, args, config, tokenizer, logits, dc, ds, prompt):
    generated = []
    for step in range(0, args.max_tokens, self.block):
      logits, tokens = self.decode(logits, dc, ds, prompt+step, step, min(self.block,args.max_tokens-step))
      for token in self.readback(tokens).reshape(-1):
        generated.append(int(token))
        if int(token) == config.traj_ids['future_start']:
          return generated
    raise RuntimeError(f'No trajectory anchor after {args.max_tokens} tokens: {tokenizer.decode(generated)}')
