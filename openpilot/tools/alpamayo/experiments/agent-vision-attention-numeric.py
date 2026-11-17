"""CPU rounding/layout investigation. This is not a native-kernel numerical attestation."""
import importlib, json, os, sys
from pathlib import Path
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,str(Path(__file__).parent))
C=importlib.import_module('agent-vision-attention-common')

def main():
  import numpy as np
  def rounded(x):
    u=x.astype(np.float32).view(np.uint32)
    return (((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint32)<<16).view(np.float32)
  def f32(x):return (x.astype(np.uint32)<<16).view(np.float32)
  records=[]
  for name in ('random','high_logits','uniform'):
    q,k,v=[f32(np.load(ROOT/f'prepared/{name}-{key}.npy',allow_pickle=False)) for key in ('q','k','v')]
    reference=np.load(ROOT/f'prepared/{name}-o.npy',allow_pickle=False)
    results={}
    for tile in (32,64):
      o=np.empty_like(q)
      for b in range(2):
        for h in range(16):
          maximum=np.full((720,1),-np.inf,np.float32);denom=np.zeros((720,1),np.float32);acc=np.zeros((720,72),np.float32)
          for start in range(0,720,tile):
            s=(q[b,h]@k[b,h,start:start+tile].T)*np.float32(72**-.5)
            nxt=np.maximum(maximum,s.max(-1,keepdims=True));p=np.exp(s-nxt,dtype=np.float32);scale=np.exp(maximum-nxt,dtype=np.float32)
            denom=denom*scale+p.sum(-1,keepdims=True,dtype=np.float32)
            acc=acc*scale+rounded(p)@v[b,h,start:start+tile];maximum=nxt
          o[b,h]=rounded(acc/denom)
      results[str(tile)]=C.metrics(o,reference)
    records.append({'fixture':name,'CPU_online_tile_results':results,
      'classification':'Approximate recurrence only; CPU matmul/reduction/exp do not reproduce WMMA/FMA/FAST_EXP2 bits'})
  # Independently detect accidental width-128 scale, causal masking, and head sharing.
  q,k,v=[f32(np.load(ROOT/f'prepared/random-{key}.npy',allow_pickle=False))[1,15] for key in ('q','k','v')]
  ref=np.load(ROOT/'prepared/random-o.npy',allow_pickle=False)[1,15]
  def apply(scores,values):
    p=np.exp(scores-scores.max(-1,keepdims=True));return (p/p.sum(-1,keepdims=True))@values
  right=(q@k.T)*np.float32(72**-.5)
  padded=(np.pad(q,((0,0),(0,56)))@np.pad(k,((0,0),(0,56))).T)*np.float32(72**-.5)
  assert np.allclose(apply(padded,v),ref,rtol=3e-5,atol=2e-6)
  wrong_scale=apply((q@k.T)*np.float32(128**-.5),v)
  causal=apply(np.where(np.arange(720)[None,:]<=np.arange(720)[:,None],right,-np.inf),v)
  wrong_head=f32(np.load(ROOT/'prepared/random-v.npy',allow_pickle=False))[1,7]
  shared=apply(right,wrong_head)
  wrong={n:float(np.sqrt(np.mean((a-ref)**2))) for n,a in [('scale_128',wrong_scale),('causal',causal),('ratio_2',shared)]}
  assert min(wrong.values())>.001
  result={'gpu_executed':False,'CPU_only':True,'recurrences':records,'wrong_semantics_rmse':wrong,
    'zero_head_padding_72_to_128_preserves_FP32_reference':True,'source_sha256':C.sha(__file__)}
  (ROOT/'cpu-rounding.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'CPU_only':True,'wrong_semantics_rmse':wrong,
    'tile_rmse':{r['fixture']:{k:v['rmse'] for k,v in r['CPU_online_tile_results'].items()} for r in records}},indent=2))
if __name__=='__main__':main()
