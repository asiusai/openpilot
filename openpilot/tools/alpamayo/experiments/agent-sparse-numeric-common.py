"""Strict provenance and independent arithmetic diagnostics; no tolerance adoption."""
import hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD=EXP/'agent-sparse-fc2-source'
CAPTURE=WORK/'agent-sparse-fc2-pathfix-capture/report.json'
FAILED=WORK/'agent-sparse-fc2-pathfix-micro/report.json'
PYTHONPATH='/tmp/opencode/alpamayo-asius-20261114/tinygrad:'+str(WORK/'activation-sparsity-devicefix-source/alpamayo/experiments')+':'+str(WORK/'activation-sparsity-devicefix-source/alpamayo')


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def local(name):return load(HERE/(name+'.py'),name.replace('-','_'))
def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb')as stream:
    while value:=stream.read(8*1024*1024):h.update(value)
  return h.hexdigest()
def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,data):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def old():return load(OLD/'agent-sparse-fc2-common.py','numeric_frozen_common')


def capture():
  C=old();plan=json.loads((OLD/'agent-sparse-fc2-plan.json').read_text())
  overlay,runner,root=C.verify_plan(plan)
  failure=json.loads(FAILED.read_text())
  if failure.get('status')!='NUMERICAL_MISMATCH_STOPPED' or failure.get('arithmetic_exact')is not False or failure.get('timings')!=[]:
    raise ValueError('Require preserved failed exactness gate, with no timing')
  micro=load(OLD/'agent-sparse-fc2-micro.py','numeric_frozen_loader')
  manifest,records=micro.load_capture(CAPTURE,plan)
  return C,plan,manifest,records,runner,root


def dependencies():
  paths=[*HERE.glob('agent-sparse-numeric-*.py'),HERE/'agent-sparse-numeric-cpu.json',CAPTURE,FAILED,
    WORK/'agent-sparse-fc2-pathfix-capture/invocation.json',WORK/'agent-sparse-fc2-pathfix-micro/invocation.json',
    Path('/tmp/opencode/queue-alpamayo-sparse-fc2-pathfix.py'),OLD/'agent-sparse-fc2-plan.json']
  paths.extend(p for p in(HERE/'agent-sparse-numeric-cpu-compiled').glob('*')if p.is_file())
  # Bind the actual pinned renderer, scheduler, compiler, JIT and queue code.
  paths.extend(Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad/tinygrad').rglob('*.py'))
  return {str(p):sha(p)for p in sorted(paths)}


def prepare():
  cpu=json.loads((HERE/'agent-sparse-numeric-cpu.json').read_text())
  if not cpu.get('all_pass') or not cpu.get('cpu_only') or cpu.get('source_hashes')!={p.name:sha(p)for p in HERE.glob('agent-sparse-numeric-*.py')}:
    raise ValueError('Require matching completed CPU/compiler/mapping qualification')
  _,plan,manifest,records,_,_=capture()
  result=dict(schema='agent-sparse-numeric-v1',sources=dependencies(),old_plan=plan,old_capture_sha256=sha(CAPTURE),
    old_stopped_micro_sha256=sha(FAILED),shape=[2112,2048,9216],layers=[0,14,27],sample=manifest['sample'],
    cpu_reference=cpu['actual_references'],classification='NUMERICAL_DIAGNOSTIC_ONLY',
    policy='Dense aligned controls must be bitwise original. Sparse differences remain reported; no quality or adoption tolerance.',
    timing_allowed=False,route_allowed=False,holdout_used=False)
  result['identity_sha256']=digest(result);return result


def verify(plan):
  if plan['schema']!='agent-sparse-numeric-v1' or digest({k:v for k,v in plan.items()if k!='identity_sha256'})!=plan['identity_sha256']:
    raise ValueError('Numeric diagnostic plan mismatch')
  for path,expected in plan['sources'].items():
    if sha(path)!=expected:raise ValueError('Numeric diagnostic source/evidence changed: '+path)
  if str(Path(__file__).resolve())not in plan['sources']:raise ValueError('Run the frozen numeric diagnostic source')
  C,old_plan,manifest,records,runner,root=capture()
  if old_plan!=plan['old_plan'] or sha(CAPTURE)!=plan['old_capture_sha256']:raise ValueError('Captured original provenance changed')
  return C,manifest,records,runner,root


def exact_reference(a,w):
  """All finite FP8 products/partial sums are exact FP64 for K <= 9216.

  E4M3 is an integer multiple of 2^-9, with |value| <= 448. In units 2^-18,
  K*448^2*2^18 <= 4.85e14 < 2^53. Every partial sum obeys this bound.
  """
  C=old();aa=C.fmt().decode(a);ww=C.fmt().decode(w)
  if not np.isfinite(aa).all()or not np.isfinite(ww).all()or a.shape[1]>9216:raise ValueError('Reference finite/K bound violated')
  dot=aa@ww.T;absolute=np.abs(aa)@np.abs(ww).T
  # Independent int64 complete-K checks at fixed coordinates, never selected by error.
  rows=sorted(set([0,1,a.shape[0]//2,a.shape[0]-1]));cols=sorted(set([0,1,w.shape[0]//2,w.shape[0]-1]))
  ia=np.rint(aa[rows]*512).astype(np.int64);iw=np.rint(ww[cols]*512).astype(np.int64)
  expected=(ia@iw.T).astype(np.float64)*2.**-18
  np.testing.assert_array_equal(dot[np.ix_(rows,cols)],expected)
  units_bound=a.shape[1]*448**2*2**18
  assert units_bound<2**53 and np.max(absolute)*2**18<=units_bound
  return dot,absolute,dict(dot_sha256=array_sha(dot),absolute_sha256=array_sha(absolute),output_shape=list(dot.shape),
    exact_FP64_dyadic_proof=True,maximum_sum_absolute_integer_units=units_bound,
    independent_int64_complete_K_coordinates=[[r,c]for r in rows for c in cols])


def consistency(actual,dot,absolute,k,seed=None):
  """Conditional RNE FP32 bound; NEVER an adoption or model quality tolerance."""
  n=k+1;u=2.**-24;gamma=n*u/(1-n*u)
  c=np.zeros_like(dot)if seed is None else np.asarray(seed,np.float64)
  reference=dot+c;bound=gamma*(absolute+np.abs(c))+n*2.**-150
  error=np.abs(np.asarray(actual,np.float64)-reference)
  finite=np.isfinite(actual)
  return dict(rule='Conditional gamma_(K+1)*sum_abs + (K+1)*2^-150; diagnostic consistency, NEVER adoption or route tolerance',
    K=k,gamma=gamma,elements_outside_bound=int(((error>bound)|~finite).sum()),maximum_absolute_error=float(error.max())if finite.all()else None,
    maximum_bound=float(bound.max()),all_finite=bool(np.isfinite(actual).all()),no_quality_claim=True)


def values(array):
  a=np.asarray(array)
  return(a.astype(np.uint32)<<16).view(np.float32)if a.dtype==np.uint16 else a


def metrics(actual,reference,worst=8):
  """Full-array bit and value evidence. No threshold selects or accepts a model."""
  a,b=np.asarray(actual),np.asarray(reference)
  if a.shape!=b.shape:raise ValueError('Metric shape mismatch')
  av,bv=values(a).astype(np.float64),values(b).astype(np.float64)
  finite=np.isfinite(av)&np.isfinite(bv);error=np.where(finite,np.abs(av-bv),np.inf)
  finite_error=error[finite];bf=bool(finite.all())
  result=dict(elements=int(a.size),actual_nonfinite=int((~np.isfinite(av)).sum()),reference_nonfinite=int((~np.isfinite(bv)).sum()),
    all_finite=bf,maximum_absolute_error=float(finite_error.max())if finite_error.size else None,
    mean_absolute_error=float(finite_error.mean())if finite_error.size else None,
    rmse=float(np.sqrt(np.mean(finite_error**2)))if finite_error.size else None,
    reference_maximum_absolute=float(np.max(np.abs(bv[finite])))if finite.any()else None,
    relative_L2=float(np.linalg.norm((av-bv)[finite])/max(np.linalg.norm(bv[finite]),np.finfo(np.float64).tiny))if finite.any()else None)
  ulp=None;diff=None
  if a.dtype==b.dtype and a.dtype in(np.dtype('float32'),np.dtype('uint16')):
    typ=np.uint32 if a.dtype==np.float32 else np.uint16;sign=1<<(31 if typ==np.uint32 else 15)
    ab,bb=a.view(typ),b.view(typ);diff=ab!=bb
    ao=np.where(ab&sign,np.bitwise_not(ab),ab|sign).astype(np.uint64)
    bo=np.where(bb&sign,np.bitwise_not(bb),bb|sign).astype(np.uint64)
    ulp=np.maximum(ao,bo)-np.minimum(ao,bo)
    result.update(bits_exact=bool(not diff.any()),differing_bits_elements=int(diff.sum()),maximum_ULP=int(ulp.max()),
      differing_signed_zeros=int(np.count_nonzero(diff&(av==0)&(bv==0))))
  count=min(worst,a.size);selected=np.argpartition(error.ravel(),a.size-count)[-count:]
  selected=selected[np.argsort(error.ravel()[selected],kind='stable')[::-1]]
  def number(v):return float(v)if np.isfinite(v)else str(v)
  result['worst_values']=[dict(coordinate=list(map(int,np.unravel_index(int(i),a.shape))),actual=number(av.ravel()[i]),
    reference=number(bv.ravel()[i]),absolute_error=number(error.ravel()[i]),
    **({}if ulp is None else dict(ULP=int(ulp.ravel()[i]),actual_bits=hex(int(a.view(typ).ravel()[i])),reference_bits=hex(int(b.view(typ).ravel()[i])))))for i in selected]
  return result


def preservation(q,values,metadata,flags):
  """Vectorized independent sparse expansion; no nonzero may be removed."""
  mt,kt=q.shape[0]//16,q.shape[1]//32
  shifts=np.arange(4,dtype=np.uint32)*4
  positions=(metadata[...,None,None]>>(shifts[None,None,None,:,None]+np.array([0,2],np.uint32)))&3
  assert np.all(positions[...,0]<positions[...,1])
  expanded=np.zeros((mt,kt,32,4,4),np.uint8)
  np.put_along_axis(expanded,positions.astype(np.intp),values.reshape(mt,kt,32,4,2),axis=-1)
  logical=expanded.reshape(mt,kt,2,16,4,4).transpose(0,3,1,2,4,5).reshape(mt,16,kt,32)
  restored=np.where(flags[:,None,:,None],logical,q.reshape(mt,16,kt,32)).reshape(q.shape)
  nonzero=(q&127)!=0
  np.testing.assert_array_equal(restored[nonzero],q[nonzero])
  np.testing.assert_array_equal(restored&127,q&127)
  return dict(all_nonzero_bytes_preserved=True,nonzero_bytes=int(nonzero.sum()),
    omitted_negative_zeros=int(np.count_nonzero(restored!=q)),signed_zero_hardware_gate_required=True)


def fixtures():
  C=old();yield from C.fixtures()
  rng=np.random.default_rng(17112026)
  values=np.array([0,56,64,68,72,184,192,196,200],np.uint8)
  for name in ('asymmetric-dense-integer','asymmetric-sparse-integer','basis-every-K-lane'):
    a=rng.choice(values,(64,64));w=rng.choice(values,(64,64))
    if name=='asymmetric-sparse-integer':
      for r in range(64):
        for g in range(16):
          positions=rng.permutation(4)[2:];a[r,g*4+positions]=0
    if name=='basis-every-K-lane':
      a[:]=0;a[np.arange(64),np.arange(64)]=56
    seed=np.zeros((64,64),np.float32)
    yield dict(name=name,a=a,w=w,c=seed,xs=np.array([.071234569],np.float32),ws=np.array([.01933221],np.float32),integer_control=True)
