"""CPU-only K128/K256 numerical probe using original BF16 fc2 masters.

No route inputs, learned transforms, clipping search, fitting, or training.
Original FP8 cache weights are a separate reference, never quantization masters.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def identity(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bf16(x):
  x=np.asarray(x,dtype=np.float32);u=x.view(np.uint32)
  return (((u+0x7fff+((u>>16)&1))>>16).astype(np.uint32)<<16).view(np.float32)


def fp8_lut():
  u=np.arange(256,dtype=np.uint8);e=(u>>3)&15;m=u&7
  out=np.where(e==0,m.astype(np.float32)/512,(1+m.astype(np.float32)/8)*np.exp2(e.astype(np.float32)-7))
  out=np.where(u&128,-out,out).astype(np.float32);out[(e==15)&(m==7)]=np.nan
  return out


def fp8_codes(x,scale):
  levels=fp8_lut()[:127]
  target=np.minimum(np.abs(np.divide(x,np.float32(scale),dtype=np.float32)),np.float32(448))
  hi=np.clip(np.searchsorted(levels,target),1,126);lo=hi-1
  dl,du=target-levels[lo],levels[hi]-target
  code=np.where((dl<du)|((dl==du)&((lo&1)==0)),lo,hi).astype(np.uint8)
  return code | (np.signbit(x).astype(np.uint8)<<7)


def quantize(x,group,signed=False):
  """One group at a time; group-major FP32 scales [groups,rows]."""
  x=np.asarray(x,dtype=np.float32);rows,k=x.shape
  if k%group:raise ValueError('Group size must divide K')
  codes=np.empty_like(x,dtype=np.int8 if signed else np.uint8);scales=np.empty((k//group,rows),np.float32)
  for g,start in enumerate(range(0,k,group)):
    block=x[:,start:start+group]
    maximum=np.max(np.abs(block) if signed else block,axis=1)
    if signed:scale=np.maximum(maximum/np.float32(7),np.float32(1e-12))
    else:scale=np.where(maximum==0,np.float32(1),maximum/np.float32(15)).astype(np.float32)
    codes[:,start:start+group]=np.clip(np.rint(np.divide(block,scale[:,None],dtype=np.float32)),-7 if signed else 0,7 if signed else 15)
    scales[g]=scale
  return codes,scales


def dequantize(q,s,group):
  out=np.empty_like(q,dtype=np.float32)
  for g,start in enumerate(range(0,q.shape[1],group)):
    out[:,start:start+group]=q[:,start:start+group].astype(np.float32)*s[g,:,None]
  return out


def grouped_dot(a,w,sa,sw,group,control=False):
  """GEMM contract: exact group INT32 -> FP32 scale product -> mul -> add.

  FP32 BLAS computes exact small integer dots because all partial sums fit 2**24.
  Control mode independently checks every group against INT64 matrix products.
  Only one group's dot and scale-product matrices are live at a time.
  """
  m,n=len(a),len(w);out=np.zeros((m,n),np.float32);ideal=np.zeros((m,n),np.float64)
  rounded_scale_ideal=np.zeros_like(ideal)
  for g,start in enumerate(range(0,a.shape[1],group)):
    aa=a[:,start:start+group];ww=w[:,start:start+group]
    dot=aa.astype(np.float32)@ww.astype(np.float32).T
    if control:np.testing.assert_array_equal(dot,aa.astype(np.int64)@ww.astype(np.int64).T)
    assert 120*group<2**24
    scale=np.multiply(sa[g,:,None],sw[g,None,:],dtype=np.float32)
    term=np.multiply(dot,scale,dtype=np.float32)
    out=np.add(out,term,dtype=np.float32)
    ideal+=dot.astype(np.float64)*(sa[g,:,None].astype(np.float64)*sw[g,None,:].astype(np.float64))
    rounded_scale_ideal+=dot.astype(np.float64)*scale.astype(np.float64)
  return out,ideal,rounded_scale_ideal


def relative(actual,expected):
  a,e=np.asarray(actual,np.float64),np.asarray(expected,np.float64)
  den=float(np.linalg.norm(e));return float(np.linalg.norm(a-e)/max(den,1e-300))


def controls():
  rng=np.random.default_rng(128256);records=[]
  for group in (128,256):
    a=rng.integers(0,16,(17,512),dtype=np.int32);w=rng.integers(-8,8,(48,512),dtype=np.int32)
    sa=np.ones((512//group,17),np.float32);sw=np.ones((512//group,48),np.float32)
    out,ideal,_=grouped_dot(a,w,sa,sw,group,control=True)
    # Independent scalar dot checks, including A=15 and W=-8.
    for row,col in ((0,0),(16,47),(7,15)):
      exact=sum(int(a[row,k])*int(w[col,k]) for k in range(512))
      assert int(out[row,col])==exact and int(ideal[row,col])==exact
    for av,wv in ((15,-8),(15,7),(0,-8)):
      aa=np.full((3,512),av,np.int32);ww=np.full((5,512),wv,np.int32)
      got,ref,_=grouped_dot(aa,ww,sa[:,:3],sw[:,:5],group,control=True)
      np.testing.assert_array_equal(got,av*wv*512);np.testing.assert_array_equal(ref,av*wv*512)
    # Non-power-of-two scale products and cancellation make FP32 order observable.
    sa=rng.uniform(.01,2,sa.shape).astype(np.float32);sw=rng.uniform(.01,2,sw.shape).astype(np.float32)
    got,ideal,rounded=grouped_dot(a,w,sa,sw,group,control=True)
    scalar=np.zeros_like(got)
    for r in range(17):
      for c in range(48):
        v=np.float32(0)
        for g,start in enumerate(range(0,512,group)):
          dot=sum(int(a[r,k])*int(w[c,k]) for k in range(start,start+group))
          scale=np.float32(sa[g,r]*sw[g,c]);term=np.float32(np.float32(dot)*scale);v=np.float32(v+term)
        scalar[r,c]=v
    np.testing.assert_array_equal(got.view(np.uint32),scalar.view(np.uint32))
    x=np.zeros((17,512),np.float32);x[1,:group]=15;x[2,group:]=np.float32(2**-133)
    x[3,:group]=np.arange(group,dtype=np.float32)%16;x[4,group+1]=65536
    qa,xa=quantize(x,group);qw,xw=quantize(np.zeros((48,512),np.float32),group,True)
    assert np.all(qa[0]==0) and np.all(xa[:,0]==1) and np.all(qw==0) and np.all(xw==np.float32(1e-12))
    assert qa[2,group:].min()==15 and np.all(xa[1:,2]>0)
    zero,zero64,_=grouped_dot(qa,qw,xa,xw,group,control=True)
    np.testing.assert_array_equal(zero,0);np.testing.assert_array_equal(zero64,0)
    records.append(dict(group=group,integer_scalar_controls=True,fp32_order_exact=True,zero_and_subnormal_groups=True,
                        fp32_vs_fp64_relative=relative(got,ideal),fp32_vs_rounded_scale_fp64=relative(got,rounded),
                        group_int32_bounds=[-120*group,105*group]))
  # Independent signed E4M3FN tie and subnormal values.
  lut=fp8_lut();v=np.array([0,1/512,1.0625,1.1875,15,448,-448],np.float32)
  np.testing.assert_array_equal(lut[fp8_codes(v,1)],np.array([0,1/512,1,1.25,15,448,-448],np.float32))
  return records


def synthetic(rows,width,seed):
  rng=np.random.default_rng(seed)
  normal=bf16(np.maximum(rng.standard_normal((rows,width)).astype(np.float32),0)**2)
  yield 'relu2',normal
  spikes=normal.copy()
  for row in range(rows):spikes[row,(row*367+17)%width]=bf16(spikes[row].max()*64)
  yield 'isolated-spikes',spikes
  yield 'channel-gain',bf16(normal*np.exp2(np.linspace(-4,4,width,dtype=np.float32))[None,:])
  zeros=normal.copy().reshape(rows,-1,128);zeros[:,::3]=0;zeros[:,1]=np.float32(2**-133)
  yield 'zero-groups-subnormals',zeros.reshape(rows,width)


def cached_fp8(cache,name,master):
  meta=json.loads((cache/'index.json').read_text());records=[r for r in meta['records'] if r['name']==name]
  sr=next(r for r in records if r['kind']=='scale');wr=next(r for r in records if r['kind']=='fp8')
  scale=np.memmap(cache/'weights.bin',mode='r',dtype='<f4',offset=sr['offset'],shape=(1,))[0]
  raw=np.memmap(cache/'weights.bin',mode='r',dtype=np.uint8,offset=wr['offset'],shape=tuple(wr['shape']))
  rows,k=master.shape
  code=(raw.reshape(rows//16,k//16,4,16,4).transpose(0,3,1,2,4).reshape(rows,k).copy() if wr['packed'] else np.array(raw))
  # Check actual cache codes against BF16-master-derived original FP8 RTN.
  ii=np.linspace(0,master.size-1,4096,dtype=np.int64)
  np.testing.assert_array_equal(code.flat[ii],fp8_codes(master.flat[ii],scale))
  return fp8_lut()[code],np.float32(scale),dict(index_sha256=identity(cache/'index.json'),record=wr,
    packed_bytes_sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest(),weight_scale=float(scale),
    checked_master_code_samples=4096,cache_whole_blob_sha256_declared=meta['sha256'])


def probe(args):
  model=local('agent-int4model');headers=model.checkpoint_headers(args.checkpoint)
  results=[]
  for layer in (0,14,27):
    started=time.perf_counter();name=f'vlm.model.language_model.layers.{layer}.mlp.fc2.weight'
    info=headers[name];raw=model.read_bf16(args.checkpoint,info);master=model.decode_bf16(raw)
    n,k=master.shape
    fp8w,fp8ws,fp8_info=cached_fp8(args.fp8_cache,name,master)
    record=dict(layer=layer,name=name,shape=[n,k],master=dict(dtype=info['dtype'],shard=info['shard'],
      data_offsets=info['data_offsets'],sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest()),fp8=fp8_info,variants={})
    for group in (k,128,256):
      label='perrow-peroutput' if group==k else f'group{group}'
      qw,sw=quantize(master,group,True);wd=dequantize(qw,sw,group)
      variant=dict(group=group,weight_relative_error=relative(wd,master),cohorts={})
      for cohort,x in synthetic(args.rows,k,9060+layer):
        a,sa=quantize(x,group);xd=dequantize(a,sa,group)
        original=x.astype(np.float64)@master.astype(np.float64).T
        out,ideal,rounded=grouped_dot(a,qw,sa,sw,group)
        metrics=dict(activation_relative_error=relative(xd,x),weight_only_feature_error=relative(x.astype(np.float64)@wd.astype(np.float64).T,original),
          activation_only_feature_error=relative(xd.astype(np.float64)@master.astype(np.float64).T,original),
          joint_fp64_error=relative(ideal,original),joint_fp32_error=relative(out,original),joint_bf16_error=relative(bf16(out),original),
          fp32_reduction_vs_fp64=relative(out,ideal),fp32_vs_rounded_scale_fp64=relative(out,rounded),
          row_zero_groups=int(np.count_nonzero(np.max(x.reshape(len(x),-1,group),axis=-1)==0)),
          scale_bytes=int(sa.nbytes+sw.nbytes),packed_matrix_bytes=int((a.size+qw.size)//2),
          input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        if group==k:
          scale=np.maximum(np.max(x)/np.float32(448),np.float32(1e-12)).astype(np.float32)
          fa=fp8_lut()[fp8_codes(x,scale)]
          fdot=fa.astype(np.float64)@fp8w.astype(np.float64).T
          metrics['original_fp8_cache_dynamicA_fp64_error']=relative(fdot*(float(scale)*float(fp8ws)),original)
          # The exact FP32 WMMA accumulation order is not emulated on CPU.
          fproduct=np.float32(scale*fp8ws)
          metrics['original_fp8_cache_dynamicA_bf16_proxy_error']=relative(bf16(fdot.astype(np.float32)*fproduct),original)
          metrics['fp8_activation_scale']=float(scale)
        variant['cohorts'][cohort]=metrics
      record['variants'][label]=variant
      print(json.dumps(dict(layer=layer,variant=label,weight_error=variant['weight_relative_error'],
        feature_errors={c:v['joint_bf16_error'] for c,v in variant['cohorts'].items()})),flush=True)
      del qw,sw,wd
    record['cpu_seconds']=time.perf_counter()-started;results.append(record)
    args.output.write_text(json.dumps(dict(cpu_only=True,complete=False,layers=results),indent=2)+'\n')
    del master,raw,fp8w
  return dict(cpu_only=True,complete=True,synthetic_only=True,route_samples_read=False,training=False,groups=[128,256],
    rows_per_cohort=args.rows,controls=controls(),checkpoint=str(args.checkpoint),
    checkpoint_config_sha256=identity(args.checkpoint/'config.json'),checkpoint_index_sha256=identity(args.checkpoint/'model.safetensors.index.json'),
    sources={str(Path(__file__)):identity(__file__),str(Path(__file__).with_name('agent-int4model.py')):identity(Path(__file__).with_name('agent-int4model.py'))},
    original_fp8_reference='Unmodified FP8 cache weight bytes; original model.py tensor max/448 dynamic synthetic activation scaling; FP64 accumulation.',
    fp8_rounding_limit='BF16 proxy rounds FP64 dot to FP32 once. It does not reproduce the target WMMA reduction order or fixed-prefill activation calibration.',
    group_contract='INT32 partial per group; FP32(sa*sw); FP32(int32*scale); separate FP32 increasing-group sum; final BF16 RNE.',layers=results)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=('controls','probe'),default='controls')
  p.add_argument('--checkpoint',type=Path,default=Path('/tmp/opencode/alpamayo-asius-20261114/model'))
  p.add_argument('--fp8-cache',type=Path,default=Path('/dev/shm/alpamayo-9060-fp8'))
  p.add_argument('--rows',type=int,default=24);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  if args.rows!=24:p.error('This bounded probe fixes 24 independently synthesized rows per cohort')
  report=dict(cpu_only=True,controls=controls()) if args.suite=='controls' else probe(args)
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
