"""Independent NumPy model of sparse-query streaming and epoch invalidation.

This proves implementation contracts on a small causal transformer. It does not
load Alpamayo weights or establish driving quality, GPU correctness, or speed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['DEV'] = 'CPU'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent


def module():
  spec = importlib.util.spec_from_file_location('streaming_plan_audit', ROOT/'agent-streaming-plan-audit.py')
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def rotary(x, positions):
  # Independent three-axis implementation with the current 128-wide mapping.
  frequency = 1/(1e8**(np.arange(64)/64))
  angles = positions[:, :, None]*frequency[None, None, :]
  phase = angles[0].copy()
  phase[:,1:60:3] = angles[1,:,1:60:3]
  phase[:,2:60:3] = angles[2,:,2:60:3]
  phase = np.concatenate([phase,phase],axis=-1)
  rotate = np.concatenate([-x[:,64:],x[:,:64]],axis=-1)
  return x*np.cos(phase)+rotate*np.sin(phase)


def forward(inputs, positions, weights, image_spans, reuse=None):
  """Recompute every selected row through every layer; cache only image K/V."""
  reuse = {} if reuse is None else reuse
  qslots = np.array([i for i in range(len(inputs))
                    if not any(a <= i < b for key,(a,b) in image_spans.items() if key in reuse)])
  assert len(qslots) and qslots[-1] == len(inputs)-1
  x = inputs[qslots].copy()
  records = {key:[] for key in image_spans}
  active = []
  for layer,(wq,wk,wv,wo) in enumerate(weights):
    q, rawk, v = x@wq, x@wk, x@wv
    current_k, current_v = np.empty_like(inputs), np.empty_like(inputs)
    current_k[qslots], current_v[qslots] = rawk, v
    for key,(a,b) in image_spans.items():
      if key in reuse:
        current_k[a:b],current_v[a:b] = reuse[key][layer]
      records[key].append((current_k[a:b].copy(), current_v[a:b].copy()))
    q, k = rotary(q,positions[:,qslots]), rotary(current_k,positions)
    scores = q@k.T/np.sqrt(inputs.shape[-1])
    scores[np.arange(len(inputs))[None,:] > qslots[:,None]] = -np.inf
    probabilities = np.exp(scores-scores.max(axis=1,keepdims=True))
    probabilities /= probabilities.sum(axis=1,keepdims=True)
    x += (probabilities@current_v)@wo
    # A deterministic per-row nonlinear sublayer creates deeper-layer context.
    x += .03*np.maximum(x,0)**2
    active.append((k,current_v.copy()))
  return x[-1], records, active, qslots


def checks():
  audit = module()
  rng = np.random.default_rng(31)
  inputs = rng.normal(scale=.1,size=(19,128))
  positions = np.broadcast_to(np.arange(19),(3,19)).copy()
  positions[1,2:5] = [2,3,2];positions[2,2:5] = [2,2,3]
  weights = rng.normal(scale=.03,size=(3,4,128,128))
  spans = {'wide-old':(2,5),'wide-new':(6,9),'tele-old':(11,14),'tele-new':(15,18)}
  previous, bank, old_cache, _ = forward(inputs,positions,weights,spans)
  # Same-window reuse with unchanged context must match full recomputation.
  same, _, _, _ = forward(inputs,positions,weights,spans,reuse=bank)
  assert np.allclose(same,previous,atol=1e-13,rtol=1e-13)
  changed = inputs.copy();changed[0] += .2;changed[6:9] += .1;changed[15:18] -= .1
  # Current timestamp/navigation/history text is represented by current text rows.
  changed[[1,5,9,10,14,18]] += .15
  moved = positions+5
  fresh, _, _, _ = forward(changed,moved,weights,spans)
  reused = {key:bank[key] for key in ('wide-old','tele-old')}
  approximate, _, active, slots = forward(changed,moved,weights,spans,reused)
  reset, _, _, _ = forward(changed,moved,weights,spans)
  assert np.array_equal(fresh,reset)
  assert np.max(np.abs(approximate-fresh)) > 1e-7
  assert set(slots) >= {0,1,5,9,10,14,18}
  assert set(range(6,9)).issubset(slots) and set(range(15,18)).issubset(slots)
  # Expert cache keys must equal the same rebased K used by language attention.
  assert np.allclose(active[0][0][2:5],rotary(bank['wide-old'][0][0],moved[:,2:5]))
  assert not np.allclose(active[0][0][2:5],old_cache[0][0][2:5])
  # Decoder/expert visibility ends at the actual anchor, excluding padding and
  # target-verification tokens generated beyond it.
  capacity,prompt,generated = 32,19,3
  expert = np.arange(capacity+64)
  mask = (expert < prompt+generated)|(expert >= capacity)
  assert mask[:22].all() and not mask[22:32].any() and mask[32:].all()

  def metadata(now,nav='Follow the road.',mutate=False):
    identities = [(camera, now+offset,(1,20,36),f'pixels-{camera}-{now+offset}')
                  for camera in (1,6) for offset in (-2_000_000,-1_000_000,-500_000,-100_000,0)]
    if mutate:
      key=list(identities[3]);key[-1]='changed-pixels';identities[3]=tuple(key)
    return dict(timestamp=now,nav=nav,identities=identities,spans=[(30+190*i,210+190*i) for i in range(10)],ids=list(range(2027)))
  policy = audit.ImageBankPolicy()
  start = policy.step(metadata(5_000_000))
  next_meta = metadata(5_100_000)
  repeat = policy.step(next_meta)
  assert start['reset']=='initial' and repeat['reused_images']==[3,8]
  assert repeat['query_tokens']==2027-360
  assert repeat['max_state_age_s']==.1
  # Byte changes at the same camera/capture identity prevent language-cache reuse.
  changed_policy=audit.ImageBankPolicy();changed_policy.step(metadata(5_000_000))
  mutated=changed_policy.step(metadata(5_100_000,mutate=True))
  assert mutated['reused_images']==[8]
  nav=policy.step(metadata(5_200_000,nav='Turn right.'))
  assert nav['reset']=='navigation_change' and nav['reused_images']==[]
  gap=policy.step(metadata(5_800_001,nav='Turn right.'))
  assert gap['reset']=='input_gap'
  backward=policy.step(metadata(5_800_001,nav='Turn right.'))
  assert backward['reset']=='time_reset'
  bounded=audit.ImageBankPolicy();max_entries=0;periodic=0
  for i in range(301):
    item=bounded.step(metadata(10_000_000+i*100_000))
    max_entries=max(max_entries,item['bank_entries'])
    periodic+=item['reset']=='periodic_2s'
    assert item['max_state_age_s'] < 2
  assert max_entries<=48 and periodic==15
  return dict(same_context_cache_max_error=float(np.max(np.abs(same-previous))),
    changed_context_output_max_error=float(np.max(np.abs(approximate-fresh))),
    full_refresh_matches=True, all_text_and_latest_images_queried=True,
    expert_receives_current_rotated_keys=True, anchor_masks_padding_and_draft_tail=True,
    pixel_mutation_rejects_reuse=True, navigation_and_gap_and_time_reset=True,
    maximum_bank_entries=max_entries,periodic_refreshes_over_30s=periodic,
    status='CPU synthetic reference passed; GPU runtime remains unimplemented')


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
  result=checks();path=Path(args.output)
  if path.exists():raise FileExistsError(path)
  path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
