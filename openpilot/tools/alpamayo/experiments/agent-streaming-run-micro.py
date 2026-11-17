"""Parent-only actual gfx1200 correctness gate; no checkpoint or performance claim."""
import argparse,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stream_micro_common',HERE/'agent-streaming-run-common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-gpu',action='store_true')
  parser.add_argument('--dry-run',action='store_true');options=parser.parse_args()
  common.verify_sources()
  if options.dry_run:
    print('Parent micro: original selected shuffle-FMA versus original-slot queries; poisoned padding, mutable KV/maps, JIT replay.');return
  if not options.parent_gpu:parser.error('Parent only: --parent-gpu with global flock and timeout')
  common.parent_gpu_environment(require_micro=False)
  import numpy as np
  from tinygrad import Tensor,TinyJit,Device,dtypes
  from tinygrad.device import Compiled
  Compiled.wait_timeout_ms=10000
  import wgp_compiler
  wgp_compiler.install(Device[Device.DEFAULT])
  if Device[Device.DEFAULT].arch!='gfx1200':raise ValueError('Frozen RX 9060 target required')
  kernels=common.load('agent-streaming-run-kernels');rng=np.random.default_rng(1749)
  report=common.report_base('parent-gpu-micro');report.update(passed=False,cases=[],arch=Device[Device.DEFAULT].arch)
  output=common.ROOT/'micro-gpu.json'
  if output.exists():raise ValueError('Move an earlier report before rerunning')
  def gpu(a,dtype=None):return Tensor(np.asarray(a),device='CPU').to(Device.DEFAULT).cast(dtype or dtypes.float32).contiguous().realize()
  def download(t):return t.float().numpy()
  try:
    # Selected source and candidate both use the exact same accumulation family.
    for rows in (17,640,1152,1536,2048):
      baseline=TinyJit(lambda q,k,v:kernels.selected().attention(q,k,v,True,waves=8,transpose='shuffle-fma',value_layout=True).realize())
      sparse=TinyJit(lambda q,k,v,s,l:kernels.attention(q,k,v,s,l).realize())
      case=dict(rows=rows,iterations=[])
      for iteration in range(4):
        valid=2029-iteration
        q,k,v=(rng.normal(0,.4,size=s).astype(np.float32) for s in ((1,16,2048,128),(1,8,2048,128),(1,8,2048,128)))
        count=min(rows-7,valid)
        slots=np.full(rows,-1,np.int32);slots[:count]=np.sort(rng.choice(valid,size=count,replace=False))
        q_gpu,k_gpu,v_gpu=(gpu(a,dtypes.bfloat16) for a in (q,k,v))
        expected=download(baseline(q_gpu,k_gpu,v_gpu))[:,:,slots[:count]]
        s=gpu(slots,dtypes.int32);limits=gpu(np.array([valid,count],np.int32),dtypes.int32)
        qs=kernels.gather(q_gpu,s).realize()
        actual=download(sparse(qs,k_gpu,v_gpu,s,limits))
        np.testing.assert_array_equal(actual[:,:,:count],expected)
        np.testing.assert_array_equal(actual[:,:,count:],0)
        # Same valid keys, extreme poisoned padded storage, same live graph.
        k[:,:,valid:]=65536;v[:,:,valid:]=-65536
        poisoned=download(sparse(qs,gpu(k,dtypes.bfloat16),gpu(v,dtypes.bfloat16),s,limits))
        np.testing.assert_array_equal(poisoned,actual)
        # Mutation after an early original slot cannot affect earlier queries.
        cutoff=valid//2;k[:,:,cutoff:]=1024;v[:,:,cutoff:]=-1024
        changed=download(sparse(qs,gpu(k,dtypes.bfloat16),gpu(v,dtypes.bfloat16),s,limits))
        early=np.flatnonzero((slots>=0)&(slots<cutoff))
        np.testing.assert_array_equal(changed[:,:,early],actual[:,:,early])
        case['iterations'].append(dict(iteration=iteration,valid=valid,queries=count,exact_selected_reference=True,
          padded_poison_unchanged=True,future_mutation_causal=True))
      baseline.reset();sparse.reset();report['cases'].append(case)
      print('Parent attention micro passed:',rows,flush=True)
    # Actual GPU gather/assemble/stage/scatter dependency and persistent JIT mutation.
    bank_np=np.arange(2*1*8*41*128,dtype=np.float32).reshape(2,1,8,41,128)%31
    bank=gpu(bank_np,dtypes.bfloat16);stage=Tensor.zeros(2,1,8,3,128,device=Device.DEFAULT,dtype=dtypes.bfloat16).realize()
    def assemble(q,qr,br,frames):
      full=kernels.assemble(q,bank,qr,br).realize();stage.assign(kernels.gather(full,frames)).realize();return full
    build=TinyJit(assemble);commit=TinyJit(lambda dest:kernels.scatter(bank,stage,dest).realize())
    for iteration in range(5):
      q=rng.normal(size=(2,1,8,5,128)).astype(np.float32)+iteration
      bits=q.view(np.uint32);q=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
      qr=np.array([0,-1,1,-1,2,3,-1,-1],np.int32);br=np.array([-1,iteration+1,-1,7,-1,-1,10,-1],np.int32)
      frames=np.array([6,1,4],np.int32);dest=np.array([7,-1,2],np.int32)
      expected=np.zeros((2,1,8,8,128),np.float32)
      for slot in range(8):
        if qr[slot]>=0:expected[...,slot,:]=q[...,qr[slot],:]
        elif br[slot]>=0:expected[...,slot,:]=bank_np[...,br[slot],:]
      actual=build(gpu(q,dtypes.bfloat16),gpu(qr,dtypes.int32),gpu(br,dtypes.int32),gpu(frames,dtypes.int32))
      np.testing.assert_array_equal(download(actual),expected)
      for i,d in enumerate(dest):
        if d>=0:bank_np[...,d,:]=expected[...,frames[i],:]
      commit(gpu(dest,dtypes.int32));np.testing.assert_array_equal(download(bank),bank_np)
    report.update(passed=True,copy_jit_replays=5,copy_dtype='BF16',
      claim='Actual synthetic micro correctness only; checkpoint control and route quality still required')
  except BaseException as error:report['failure']=repr(error);raise
  finally:output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
