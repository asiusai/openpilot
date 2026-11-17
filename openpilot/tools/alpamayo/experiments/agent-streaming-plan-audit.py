"""CPU-only prompt identity, streaming schedule, and cache-contract audit.

No model inference, training, GPU access, generated outputs, or future-motion
references. The lightweight tokenizer path is checked against actual FastInputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['DEV'] = 'CPU'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
from collections import Counter, OrderedDict
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
spec = importlib.util.spec_from_file_location('streaming_audit_inputs', ROOT/'fast-inputs.py')
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
  # Writable copy-on-write mappings cannot change the frozen fixture.
  return {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
          else torch.from_numpy(np.load(p, allow_pickle=False, mmap_mode='c'))
          for p in Path(path).glob('*.npy')}


def prefix(a, b):
  n = min(len(a), len(b))
  unequal = np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))
  return int(unequal[0]) if len(unequal) else n


def prompt(adapter, data):
  """Use the checkpoint template, tokenizer, history tokenizer, and actual mRoPE."""
  processor = adapter.processor
  canonical = dict(data, camera_pixel_bounds=adapter.pixel_bounds)
  messages = fast.reference.build_conversation(canonical)
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
    add_vision_id=False, continue_final_message=False, enable_thinking=False)
  image = processor.image_processor
  grids = []
  for frame, camera in zip(data['image_frames'], data['camera_indices'].tolist(), strict=True):
    h, w = smart_resize(*frame.shape[-2:], factor=image.patch_size*image.merge_size,
      min_pixels=adapter.pixel_bounds[camera][0], max_pixels=adapter.pixel_bounds[camera][1])
    grids.append((1, h//image.patch_size, w//image.patch_size))
  grids = torch.tensor(grids)
  expanded, _ = processor.get_text_with_replacements([text],
    images_replacements=[processor.replace_image_token({'image_grid_thw':grids}, i) for i in range(len(grids))])
  ids = adapter.tokenizer(expanded, return_tensors='pt')['input_ids']
  history = adapter.history.encode(None, None, data['ego_history_xyz'], data['ego_history_rot'])
  ids[ids == adapter.config.traj_ids['history_pad']] = (history+adapter.config.traj_ids['history_id0']).flatten()
  types = (ids == processor.image_token_id).to(torch.int64)
  pos, delta = fast.reference.Cosmos3EdgeModel.get_rope_index(adapter.rope, ids, types, image_grid_thw=grids)
  indices = np.flatnonzero(types.numpy()[0])
  spans = [(int(s[0]), int(s[-1])+1) for s in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)]
  # Hash normalized-input-equivalent uint8 pixels after the actual bicubic resize.
  # This is stricter than an unchecked capture ID, and requires no vision inference.
  identities = []
  for frame, camera, capture, grid in zip(data['image_frames'], data['camera_indices'].tolist(),
      data['absolute_timestamps'].reshape(-1).tolist(), grids.tolist(), strict=True):
    pixels = resize(frame.squeeze(0), [grid[1]*image.patch_size, grid[2]*image.patch_size], interpolation=InterpolationMode.BICUBIC)
    digest = hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()
    identities.append((camera, capture, tuple(grid), digest))
  # Equal placeholder IDs do not imply equal input embeddings.
  fingerprints = [f't:{int(i)}' for i in ids[0]]
  for (begin, end), identity in zip(spans, identities, strict=True):
    # Capture IDs select reusable records, but are not network image embeddings.
    # Only processed pixels and grid enter this embedding-identity comparison.
    fingerprints[begin:end] = [f'image:{identity[2:]}:{j}' for j in range(end-begin)]
  return dict(ids=ids.numpy()[0], positions=pos.numpy(), delta=int(delta.item()), spans=spans,
    identities=identities, fingerprints=fingerprints, nav=data['nav_text'][0],
    timestamp=max(x[1] for x in identities), grids=grids.tolist())


class ImageBankPolicy:
  """One frozen policy: 48 image records, 2 s reset, all text and latest images fresh.

  This schedules approximate language KV reuse only. All ten current images stay
  in the prompt. The GPU runtime and learned-model quality are not implemented.
  """
  def __init__(self):
    self.bank = OrderedDict()
    self.epoch = self.previous = self.nav = None

  def step(self, meta):
    now = meta['timestamp']
    reset = ('initial' if self.epoch is None else
             'time_reset' if now <= self.previous else
             'input_gap' if now-self.previous > 500_000 else
             'navigation_change' if meta['nav'] != self.nav else
             'periodic_2s' if now-self.epoch >= 2_000_000 else None)
    if reset:
      self.bank.clear()
      self.epoch = now
    newest = {camera:max(key[1] for key in meta['identities'] if key[0] == camera)
              for camera in (1, 6)}
    reuse = [i for i, key in enumerate(meta['identities'])
             if key in self.bank and key[1] < newest[key[0]]]
    reused_tokens = sum(meta['spans'][i][1]-meta['spans'][i][0] for i in reuse)
    # Decide all reuse from the preceding completed prediction, then commit.
    # Hits keep their original build timestamp; reads cannot rejuvenate stale KV.
    ages = [(now-self.bank[meta['identities'][i]])/1e6 for i in reuse]
    for i, key in enumerate(meta['identities']):
      if i not in reuse:
        self.bank[key] = now
      self.bank.move_to_end(key)
    while len(self.bank) > 48:
      self.bank.popitem(last=False)
    self.previous, self.nav = now, meta['nav']
    return dict(reset=reset, reused_images=reuse, query_tokens=len(meta['ids'])-reused_tokens,
                reused_tokens=reused_tokens, max_state_age_s=max(ages, default=0), bank_entries=len(self.bank))


def stats(values):
  return dict(min=float(min(values)), mean=float(np.mean(values)), median=float(np.median(values)), max=float(max(values)))


def audit_sequence(adapter, paths, checks):
  previous = None
  policy = ImageBankPolicy()
  prefixes, id_prefixes, adjacent, traces, lengths, first = [], [], [], [], [], None
  for index, path in enumerate(paths):
    data = load(path)
    meta = prompt(adapter, data)
    assert data['camera_indices'].tolist() == [1]*5+[6]*5
    if index in checks:
      _, _, prepared = adapter.prepare(data)
      ids, positions, delta, _ = adapter.native_inputs(prepared)
      assert np.array_equal(ids[0], meta['ids'])
      assert np.array_equal(positions, meta['positions']) and delta == meta['delta']
      assert prepared['tokenized_data']['image_grid_thw'].tolist() == meta['grids']
    if first is None:
      first = dict(path=path, prompt_tokens=len(meta['ids']), image_spans=meta['spans'],
        grids=meta['grids'], mrope_delta=meta['delta'],
        pre_image_text=adapter.tokenizer.decode(meta['ids'][:meta['spans'][0][0]].tolist()),
        after_images_text=adapter.tokenizer.decode(meta['ids'][meta['spans'][-1][1]:].tolist()),
        first_image_position_start=meta['positions'][:,0,meta['spans'][0][0]].tolist(),
        first_image_position_end=meta['positions'][:,0,meta['spans'][0][1]-1].tolist(),
        timestamps=[key[1] for key in meta['identities']],
        text_between_images=[adapter.tokenizer.decode(meta['ids'][a[1]:b[0]].tolist())
                             for a,b in zip(meta['spans'],meta['spans'][1:])])
    if previous is not None:
      assert meta['timestamp'] > previous['timestamp']
      n = prefix(previous['fingerprints'], meta['fingerprints'])
      common_positions = np.all(previous['positions'][:,:,:n] == meta['positions'][:,:,:n], axis=(0,1))
      if not common_positions.all():
        n = int(np.flatnonzero(~common_positions)[0])
      prefixes.append(n)
      id_prefixes.append(prefix(previous['ids'], meta['ids']))
      adjacent.append(sum(key in previous['identities'] for key in meta['identities']))
    result = policy.step(meta)
    traces.append(dict(sample=path, timestamp_us=meta['timestamp'], prompt_tokens=len(meta['ids']), **result))
    lengths.append(len(meta['ids']))
    previous = meta
  return dict(samples=len(paths), first_prompt=first, prompt_tokens=stats(lengths),
    identity_valid_prefix_tokens=stats(prefixes), token_ids_only_prefix_tokens=stats(id_prefixes),
    adjacent_identical_images=stats(adjacent), exact_prefix_histogram=dict(Counter(prefixes)),
    policy=dict(query_tokens=stats([t['query_tokens'] for t in traces]),
      query_fraction=float(sum(t['query_tokens'] for t in traces)/sum(lengths)),
      reused_images=stats([len(t['reused_images']) for t in traces]),
      resets=dict(Counter(t['reset'] for t in traces if t['reset'])),
      maximum_bank_entries=max(t['bank_entries'] for t in traces),
      maximum_state_age_s=max(t['max_state_age_s'] for t in traces)), traces=traces)


def contract_checks():
  """Independent numeric checks: original-slot mask and re-RoPE do not restore V."""
  rng = np.random.default_rng(19)
  n, d = 11, 128
  positions = np.broadcast_to(np.arange(n), (3,1,n)).copy()
  positions[1] += 3
  positions[2] += 7
  cos, sin = fast.reference.rotary(positions)
  cos, sin = cos[0,0], sin[0,0]
  x = rng.normal(size=(n,d)).astype(np.float32)
  rotate = lambda z: np.concatenate((-z[...,d//2:],z[...,:d//2]),axis=-1)
  old = x*cos+rotate(x)*sin
  recovered = old*cos-rotate(old)*sin
  new_cos, new_sin = fast.reference.rotary(positions+13)
  new_cos, new_sin = new_cos[0,0], new_sin[0,0]
  rebased = recovered*new_cos+rotate(recovered)*new_sin
  expected = x*new_cos+rotate(x)*new_sin
  assert np.max(np.abs(rebased-expected)) < 2e-6
  # Casting post-RoPE keys to BF16 before inversion adds avoidable loss.
  old_bf = torch.from_numpy(old).to(torch.bfloat16).float().numpy()
  inverse_bf = old_bf*cos-rotate(old_bf)*sin
  bf_error = float(np.max(np.abs(inverse_bf-x)))
  assert bf_error > 1e-4
  slots = np.array([0,1,4,7,10])
  q, k, v = rng.normal(size=(3,n,d))
  mask = np.arange(n)[None,:] <= slots[:,None]
  scores = q[slots]@k.T/np.sqrt(d)
  scores[~mask] = -np.inf
  weights = np.exp(scores-scores.max(axis=1,keepdims=True))
  result = weights@v/weights.sum(axis=1,keepdims=True)
  ref = []
  for slot in slots:
    z = q[slot]@k[:slot+1].T/np.sqrt(d)
    p = np.exp(z-z.max());ref.append(p@v[:slot+1]/p.sum())
  assert np.allclose(result, ref)
  # A compact triangular mask incorrectly treats original slot 7 as row 3.
  assert not np.array_equal(mask, np.arange(n)[None,:] <= np.arange(len(slots))[:,None])
  # Two-layer scalar attention counterexample. Layer zero K/V is input-local;
  # layer one V of an unchanged image depends on the changed preceding token.
  # All Q/K are zero, so position changes cannot explain or repair this result.
  def two_layers(inp):
    h = np.array([np.mean(inp[:i+1]) for i in range(len(inp))])
    return h, np.array([np.mean(h[:i+1]) for i in range(len(h))])
  a, _ = two_layers(np.array([0.,2.,4.]))
  b, fresh = two_layers(np.array([10.,2.,4.]))
  cached_value = a[1]
  hybrid = np.mean([b[0],cached_value,b[2]])
  assert cached_value != b[1] and hybrid != fresh[-1]
  return dict(original_slot_attention_pass=True, mrope_float32_rebase_max_error=float(np.max(np.abs(rebased-expected))),
    inverse_postrope_bf16_max_error=bf_error,
    changed_context=dict(cached_image_layer1_value=float(cached_value), current_image_layer1_value=float(b[1]),
      hybrid_output=float(hybrid), recomputed_output=float(fresh[-1])),
    limit='CPU synthetic attention and position algebra; no Alpamayo model quality or GPU kernel qualification')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--route-samples', required=True)
  parser.add_argument('--ten-hz-samples', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  adapter = fast.FastInputs(args.checkpoint, camera_profile='nvidia', allow_missing_cameras=True, direct_byte_inputs=True)
  routes = json.loads(Path(args.route_samples).read_text())
  tens = json.loads(Path(args.ten_hz_samples).read_text())
  report = dict(schema=1, execution='CPU only; no learned-model forward pass',
    policy='image-bank-48-refresh-2s-all-text-current-latest-image-current',
    contract_checks=contract_checks(),
    route_4hz=audit_sequence(adapter, routes, {0,1,len(routes)//2}),
    prepared_10hz_window=audit_sequence(adapter, tens, {0,1,len(tens)//2}))
  source_paths = [Path(__file__), ROOT/'fast-inputs.py', ROOT.parent/'model.py', ROOT.parent/'inputs.py',
    ROOT/'padded-prefill.py',ROOT/'quantize-packed.py',ROOT/'online-attention.py',ROOT/'block-decode.py',
    Path(inspect.getfile(fast.reference.build_conversation)),Path(inspect.getfile(fast.reference._tokenize_messages)),
    Path(inspect.getfile(fast.reference.Cosmos3EdgeModel)),Path(inspect.getfile(type(adapter.processor))),
    Path(args.checkpoint)/'config.json',Path(args.checkpoint)/'chat_template.jinja',
    Path(args.route_samples),Path(args.ten_hz_samples)]
  report['source_sha256'] = {str(path):sha(path) for path in source_paths}
  report['memory_bytes'] = dict(image_record_bf16_raw_k_v=28*2*8*180*128*2,
    bank48_bf16_raw_k_v=48*28*2*8*180*128*2,active2304_bf16_k_v=28*2*8*2304*128*2)
  target = Path(args.output)
  if target.exists():
    raise FileExistsError(target)
  target.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:{x:v for x,v in report[k].items() if x not in ('traces','first_prompt')}
                    for k in ('route_4hz','prepared_10hz_window')},indent=2))


if __name__ == '__main__':
  main()
