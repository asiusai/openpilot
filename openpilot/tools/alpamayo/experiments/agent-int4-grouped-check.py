"""CPU layout/dependency proof; parent-only grouped INT4 GPU microchecks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def serial_proof(linear,count,full=False,proofs=(),expected_name=None):
  """Use the shared raw/AFTER/HCQ inspector and its real queue-barrier proof."""
  names=expected_name or ('agent_int4_grouped_quantize','agent_int4_grouped')
  jit=SimpleNamespace(captured=SimpleNamespace(_linear=linear),_int4_serial_proofs=list(proofs))
  result=local('agent-int4-scaled-check').inspect_serial(jit,names,count,full)
  return dict(result,calls=count,all_adjacent_edges=True,fixed_actual_output_storage=True)


def layout_control(kernel,probe,group):
  m,k=17,512;mp=(m+15)//16*16;ng=k//group
  rng=np.random.default_rng(128256);x=probe.bf16(np.maximum(rng.standard_normal((m,k)).astype(np.float32),0)**2)
  x[0]=0;x[1,:group]=np.arange(group,dtype=np.float32)%30/2;x[1,group-1]=15
  x[2,group:]=np.float32(2**-133);x[-1,-1]=65536
  q,scale=probe.quantize(x,group)
  expected=kernel.local('agent-int4-scaled-kernel').pack_unsigned_cpu(q)
  actual=np.zeros_like(expected);words=actual.view(np.uint32).reshape(-1);written=np.zeros(words.size,np.uint8)
  native_scale=np.empty_like(scale)
  for g in range(ng):
    for row in range(mp):
      values=np.zeros((group//8,8),np.float32) if row>=m else x[row,g*group:(g+1)*group].reshape(-1,8)
      maximum=values.max(axis=1)
      for shift in (8,4,2,1) if group==128 else (16,8,4,2,1):maximum=np.maximum(maximum,maximum[np.arange(len(maximum))^shift])
      np.testing.assert_array_equal(maximum,np.max(values))
      s=np.float32(1) if maximum[0]==0 else np.float32(maximum[0]/np.float32(15))
      if row<m:native_scale[g,row]=s
      codes=np.rint(np.clip(values/s,0,15)).astype(np.uint32)
      for sub in range(group//8):
        kk=g*group+sub*8;lane=(kk%32//16)*16+row%16;reg=kk%16//8
        dest=((row//16*(k//32)+kk//32)*32+lane)*2+reg
        words[dest]=np.bitwise_or.reduce(codes[sub]<<np.arange(0,32,4,dtype=np.uint32));written[dest]+=1
  np.testing.assert_array_equal(written,1);np.testing.assert_array_equal(actual,expected)
  np.testing.assert_array_equal(native_scale.view(np.uint32),scale.view(np.uint32))
  return dict(group=group,every_packed_word_written_once=True,subgroup_maximum_and_scale_bits_exact=True,all_padding_zero=True)


def scheduled_control(kernel,group):
  from tinygrad import Tensor,dtypes
  m,n,k=17,48,512;ng=k//group
  x=Tensor.empty(m,k,dtype=dtypes.bfloat16,device='CPU');w=Tensor.empty(3,k//32,32,8,dtype=dtypes.uint8,device='CPU')
  ws=Tensor.empty(ng,n,dtype=dtypes.float32,device='CPU');q=Tensor.empty(2,k//32,32,8,dtype=dtypes.uint8,device='CPU')
  xs=Tensor.empty(ng,m,dtype=dtypes.float32,device='CPU');out=Tensor.empty(m,n,dtype=dtypes.bfloat16,device='CPU')
  for _ in range(40):
    q,xs=kernel.quantize_pack(Tensor(x.uop.after(out.uop)),group,packed_out=q,scales_out=xs)
    out=kernel.gemm(q,w,xs,ws,m,n,group,out=out)
  linear,_=out.linear_with_vars()
  from tinygrad.uop.ops import UOp,Ops
  buffers=[u for u in linear.toposort() if u.op is Ops.BUFFER]
  linear=linear.substitute({u:UOp.param(i,u.dtype,u.max_numel(),u.device) for i,u in enumerate(buffers)},walk=True)
  return serial_proof(linear,80,full=True)


def cpu_checks(kernel,probe):
  return dict(cpu_only=True,complete=True,integer_and_fp64_controls=probe.controls(),
    layouts=[layout_control(kernel,probe,g) for g in (128,256)],
    scheduled={str(g):scheduled_control(kernel,g) for g in (128,256)})


def tensor_bf16(values):
  from tinygrad import Tensor,dtypes
  bits=(np.asarray(values,np.float32).view(np.uint32)>>16).astype(np.uint16)
  return Tensor(bits).bitcast(dtypes.bfloat16).realize()


def check_codes(kernel,probe,a,w,sa,sw,group,label):
  from tinygrad import Tensor,dtypes
  scaled=kernel.local('agent-int4-scaled-kernel');m,n,k=len(a),len(w),a.shape[1]
  pa=Tensor(scaled.pack_unsigned_cpu(a)).realize();pw=Tensor(scaled.pack_weights_cpu(w).copy()).realize()
  sx,ss=Tensor(sa).realize(),Tensor(sw).realize()
  groups=list(range(k//group)) if m<128 else sorted({0,k//group//2,k//group-1})
  for index in groups:
    start=index*group
    expected=(a[:,start:start+group].astype(np.float32)@w[:,start:start+group].astype(np.float32).T).astype(np.int32)
    got=kernel.gemm(pa,pw,sx,ss,m,n,group,raw_group=index).numpy();np.testing.assert_array_equal(got,expected)
  expected,ideal,_=probe.grouped_dot(a,w,sa,sw,group,control=m<128)
  f32=kernel.gemm(pa,pw,sx,ss,m,n,group,fp32=True).numpy()
  np.testing.assert_array_equal(f32.view(np.uint32),expected.view(np.uint32))
  output=kernel.gemm(pa,pw,sx,ss,m,n,group).bitcast(dtypes.uint16).numpy()
  bits=(probe.bf16(expected).view(np.uint32)>>16).astype(np.uint16);np.testing.assert_array_equal(output,bits)
  record=dict(label=label,shape=[m,n,k],raw_int32_groups=groups,fp32_all_bits_exact=True,bf16_all_bits_exact=True,
              fp32_vs_fp64=probe.relative(expected,ideal))
  print(json.dumps(record),flush=True);return record,bits


def check_quantizer(kernel,probe,x,group):
  from tinygrad import dtypes
  q,s=probe.quantize(x,group);p,gs=kernel.quantize_pack(tensor_bf16(x),group)
  np.testing.assert_array_equal(p.numpy(),kernel.local('agent-int4-scaled-kernel').pack_unsigned_cpu(q))
  np.testing.assert_array_equal(gs.bitcast(dtypes.uint32).numpy(),s.view(np.uint32));return q,s


def benchmark(kernel,x,pw,ws,group,expected,args):
  from tinygrad import Tensor,TinyJit,Device,dtypes
  probe=local('agent-int4-grouped-probe');q,s=probe.quantize(x,group)
  xx=tensor_bf16(x);qq=Tensor(kernel.local('agent-int4-scaled-kernel').pack_unsigned_cpu(q)).realize();ss=Tensor(s).realize()
  weight=Tensor(pw.copy()).realize();sw=Tensor(ws.copy()).realize();oo=Tensor.zeros(len(x),ws.shape[1],dtype=dtypes.bfloat16).realize()
  def full(x,w,ws,q,s,out):
    for _ in range(args.runs):
      q,s=kernel.quantize_pack(Tensor(x.uop.after(out.uop)),group,packed_out=q,scales_out=s)
      out=kernel.gemm(q,w,s,ws,x.shape[0],ws.shape[1],group,out=out)
    return out.realize()
  def gemms(q,w,s,ws,out):
    for _ in range(args.runs):out=kernel.gemm(q,w,s,ws,x.shape[0],ws.shape[1],group,out=out)
    return out.realize()
  def quantizers(x,q,s):
    for _ in range(args.runs):q,s=kernel.quantize_pack(x,group,packed_out=q,scales_out=s)
    Tensor.realize(q,s);return q,s
  records={}
  inspector=local('agent-int4-scaled-check')
  for name,fn,inputs,count in [('quantize_pack',quantizers,(xx,qq,ss),args.runs),
    ('grouped_gemm',gemms,(qq,weight,ss,sw,oo),args.runs),('full_serial',full,(xx,weight,sw,qq,ss,oo),args.runs*2)]:
    jit=TinyJit(fn)
    names=(('agent_int4_grouped_quantize','agent_int4_grouped') if name=='full_serial' else
           ('agent_int4_grouped_quantize',) if name=='quantize_pack' else ('agent_int4_grouped',))
    for _ in range(3):
      if jit.cnt==1:
        with inspector.observe_serial_queue(jit,names,count,name=='full_serial'):jit(*inputs)
      else:jit(*inputs)
    proof=serial_proof(jit.captured._linear,count,full=name=='full_serial',proofs=getattr(jit,'_int4_serial_proofs',()),expected_name=names)
    if name!='quantize_pack':np.testing.assert_array_equal(jit.captured.ret.bitcast(dtypes.uint16).numpy(),expected)
    else:
      np.testing.assert_array_equal(jit.captured.ret[0].numpy(),kernel.local('agent-int4-scaled-kernel').pack_unsigned_cpu(q))
      np.testing.assert_array_equal(jit.captured.ret[1].bitcast(dtypes.uint32).numpy(),s.view(np.uint32))
    times=[]
    for _ in range(args.rounds):
      Device[Device.DEFAULT].synchronize();start=time.perf_counter();jit(*inputs);Device[Device.DEFAULT].synchronize()
      times.append((time.perf_counter()-start)*1000/args.runs)
    records[name]=dict(ms=times,median_ms=float(np.median(times)),serial_proof=proof)
  return records


def gpu_checks(kernel,probe,args):
  local('prefill-kernels').fix_dispatch_packet();group=args.group;rng=np.random.default_rng(128256);records=[]
  m,n,k=17,48,512
  a=rng.integers(0,16,(m,k),dtype=np.int32);w=rng.integers(-8,8,(n,k),dtype=np.int32)
  sa=rng.uniform(.01,2,(k//group,m)).astype(np.float32);sw=rng.uniform(.01,2,(k//group,n)).astype(np.float32)
  for label in ('original','mutate-a','mutate-w','all15-neg8','zero-groups'):
    aa,ww=a.copy(),w.copy()
    if label=='mutate-a':aa[-1,-1]=15 if aa[-1,-1]!=15 else 0
    if label=='mutate-w':ww[-1,-1]=-8 if ww[-1,-1]!=-8 else 7
    if label=='all15-neg8':aa[:]=15;ww[:]=-8
    if label=='zero-groups':aa[:,:group]=0;ww[:,group:group*2]=0
    record,_=check_codes(kernel,probe,aa,ww,sa,sw,group,label);records.append(record)
  x=list(probe.synthetic(m,k,128256))[-1][1];x[0]=0;x[-1,-1]=65536
  x[1,:group]=np.arange(group,dtype=np.float32)%30/2;x[1,group-1]=15
  check_quantizer(kernel,probe,x,group)
  report=dict(cpu_only=False,complete=True,group=group,controls=records)
  if args.suite=='actual':
    model=local('agent-int4model');headers=model.checkpoint_headers(args.checkpoint)
    name='vlm.model.language_model.layers.0.mlp.fc2.weight';raw=model.read_bf16(args.checkpoint,headers[name]);master=model.decode_bf16(raw)
    pw,sw=kernel.quantize_weights_cpu(master,group);w,sw_check=probe.quantize(master,group,True)
    np.testing.assert_array_equal(sw,sw_check)
    x=next(probe.synthetic(2112,9216,128256))[1];x[0]=0;x[1,:group]=0;x[2,:group]=np.float32(2**-133);x[3,-1]=65536
    for label in ('actual','actual-mutation'):
      if label=='actual-mutation':x=x.copy();x[-1,-1]=131072
      a,sa=check_quantizer(kernel,probe,x,group);record,expected=check_codes(kernel,probe,a,w,sa,sw,group,label);records.append(record)
    report['master']=dict(name=name,source=headers[name],sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest())
    report['timing']=benchmark(kernel,x,pw,sw,group,expected,args)
    report['timing_contract']='40 serial calls minimum; quantizer and grouped GEMM included; constant weight preparation/uploads excluded'
  report['all_pass']=True;return report


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=('cpu','small','actual'),default='cpu')
  p.add_argument('--group',type=int,choices=(128,256),default=256);p.add_argument('--runs',type=int,default=40)
  p.add_argument('--rounds',type=int,default=3);p.add_argument('--checkpoint',type=Path,default=Path('/tmp/opencode/alpamayo-asius-20261114/model'))
  p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  if not 40<=args.runs<=80 or not 1<=args.rounds<=5:p.error('Require 40..80 calls and 1..5 rounds')
  kernel,probe=local('agent-int4-grouped-kernel'),local('agent-int4-grouped-probe')
  report=cpu_checks(kernel,probe) if args.suite=='cpu' else gpu_checks(kernel,probe,args)
  files=[Path(__file__),Path(__file__).with_name('agent-int4-grouped-kernel.py'),Path(__file__).with_name('agent-int4-grouped-probe.py'),
         Path(__file__).with_name('agent-int4-scaled-kernel.py'),Path(__file__).with_name('agent-int4-scaled-check.py'),
         Path(__file__).with_name('agent-int4model.py')]
  report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cpu_only=report['cpu_only'],output=str(args.output))))


if __name__=='__main__':main()
