"""CPU composition imports, owner/slot dispatch, cleanup, and strict rejection checks."""
import argparse
import copy
from contextlib import ExitStack
import importlib.util
import inspect
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_combined_check_common',HERE/'agent-expert-combined-common.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def main():
  parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  c,manifest = C.base(); P,evidence = C.pv(); C.paths()
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.engine.realize import _get_call_to_compile
  from tinygrad.helpers import PARALLEL
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops, UOp
  assert not Device._opened_devices
  installer = C.load(HERE/'agent-expert-combined-install.py','expert_combined_cpu_installer')
  Composition = installer.factory(C)
  fixtures = C.load(C.MLP/'experiments/agent-fused-mlp-check.py','expert_combined_cpu_fixtures')
  Fake = fixtures.FakeTensor
  renderer = P.usb().renderers()['parent-wgp']
  class FakeDevice:
    DEFAULT = 'USB+AMD:LLVM'
    def __class_getitem__(cls,key): return SimpleNamespace(renderer=renderer)
  rejected,arms = [],[]
  def reject(label,fn):
    try: fn()
    except (AssertionError,ValueError,RuntimeError,KeyError,StopIteration): rejected.append(label)
    else: raise AssertionError('Invalid control passed: '+label)
  fixed = {f'vlm.model.language_model.layers.{i}/{j}':.0037*(1+i*4+j) for i in range(28) for j in range(4)}
  shapes = P.usb().load('agent-expert-pv-compile').specs(P.SHAPE)
  buffers = [UOp.placeholder(s,dtypes.bfloat16,i) for i,s in enumerate(shapes)]
  for candidate in (False,True):
    mlps,visions,quants = [],[],[]
    def dispatch(x,w,xs,ws,output_scale,rows,variant):
      assert rows == 2048 and variant == 'transpose4x4'
      mlps.append((xs,ws,output_scale)); return Fake(shape=(2048,9216),dtype=dtypes.fp8e4m3)
    def pack(x): return Fake(shape=x.shape,dtype=x.dtype,tag='packed:'+x.tag)
    def quantize(x,scale=None):
      assert type(x) is Fake
      quants.append((x,scale)); return Fake(shape=(x.numel()//x.shape[-1],x.shape[-1]),dtype=dtypes.fp8e4m3),scale
    def vision_gemm(x,w,scales,bias,tile):
      visions.append((x.shape,scales,bias,tile)); return Fake(shape=(*x.shape[:-1],w.shape[0]),dtype=dtypes.bfloat16)
    comp = Composition('native',candidate,evidence,device_factory=lambda:SimpleNamespace(arch='gfx1200',device='AMD'),mlp_dispatch=dispatch)
    original_spec = importlib.util.spec_from_file_location
    original_queues = hcq2._build_queues
    global_before = renderer,renderer.compiler,PARALLEL.value,dict(to_program_cache)
    with ExitStack() as mocks:
      mocks.enter_context(patch.object(Fake,'ndim',property(lambda self:len(self.shape)),create=True))
      mocks.enter_context(patch.object(comp.vision,'Tensor',Fake)); mocks.enter_context(patch.object(comp.vision,'gemm',vision_gemm))
      try:
        with comp:
          reject('nested/'+str(candidate),lambda:Composition('quality',candidate,evidence).__enter__())
          core = C.load(C.MLP/'model.py','expert_combined_cpu_core')
          mocks.enter_context(patch.object(core,'Tensor',Fake))
          model = fixtures.FakeModel(core)
          model.output_token_mapping = Fake(shape=(256,),dtype=dtypes.int32)
          model.config['expert_config'] = {'llm_config':{'num_hidden_layers':28}}
          model.trajectory_jits = {}; model.cache = [Fake(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16)]*28
          for name in ('trajectory','_trajectory','expert_step'): setattr(model,name,MethodType(getattr(core.EdgeModel,name),model))
          for i in range(1,28):
            for key,value in list(model.weights.items()):
              if not key.startswith('expert.expert.layers.0.'): continue
              new = key.replace('layers.0.','layers.'+str(i)+'.')
              model.weights[new] = Fake(shape=value.shape,dtype=value.dtype,tag=new)
              model.weight_scales[new] = Fake([.125],dtype=dtypes.float32)
          model.norm = MethodType(lambda self,x,name,rms=False,eps=None:x,model)
          model.encode_jits = {}; model.encode = lambda *a:None
          expert = C.load(C.MLP/'experiments/expert-kernels.py','expert_combined_cpu_expert')
          mocks.enter_context(patch.object(expert,'pack',pack)); mocks.enter_context(patch.object(expert,'unpack',lambda x:x))
          mocks.enter_context(patch.object(expert,'gemm',lambda q,w,scales,bias,dtype:Fake(shape=(64,w.shape[0]),dtype=dtype)))
          with patch('tinygrad.Device',FakeDevice): expert.install(model)
          h = comp.pv_handle; actual_matmul,original_trajectory = h.original_matmul,h.originals.copy()
          assert global_before == (renderer,renderer.compiler,PARALLEL.value,dict(to_program_cache))
          assert model.text_layer.__func__ is core.EdgeModel.text_layer
          reject('duplicate expert/'+str(candidate),lambda:expert.install(model))
          for i in range(27):
            for fc,shape in (('fc1',(4304,1152)),('fc2',(1152,4304))):
              name = f'vlm.model.visual.encoder.layers.{i}.mlp.{fc}'
              model.weights[name+'.weight'] = Fake(shape=shape,dtype=dtypes.bfloat16 if i<2 else dtypes.fp8e4m3,tag=name)
              model.weights[name+'.bias'] = Fake(shape=(shape[0],),dtype=dtypes.bfloat16)
              if i>=2: model.weight_scales[name+'.weight'] = Fake([.125],dtype=dtypes.float32)
          model.weights['vlm.model.visual.embeddings.patch_embedding.weight'] = Fake(shape=(1152,1176),dtype=dtypes.bfloat16)
          model.grouped_quantize = lambda x,images:(Fake(shape=(images*720,x.shape[-1]),dtype=dtypes.fp8e4m3),
            Fake(shape=(images*720,),dtype=dtypes.float32))
          packed = C.load(C.MLP/'experiments/quantize-packed.py','expert_combined_cpu_packed')
          mocks.enter_context(patch.object(packed,'Tensor',Fake)); mocks.enter_context(patch.object(packed,'pack',pack))
          mocks.enter_context(patch.object(packed,'quantize',quantize)); packed.install(model,fixed_scales=fixed)
          batched = C.load(C.MLP/'experiments/batched-vision.py','expert_combined_cpu_batched'); batched.install(model)
          cachemod = C.load(C.MLP/'experiments/vision-cache.py','expert_combined_cpu_cache')
          cache = cachemod.VisionCache(model,grouped=True,group_limit=2)
          decoder = C.load(C.MLP/'experiments/lookup-decode.py','expert_combined_cpu_lookup').LookupDecoder(model)
          assert decoder.history == [] and comp.lookup_initial[0]['vision_cache_empty']
          session = comp.session; state = {'prefill':False,'prefix':'outer','index':19}; session.state.update(state)
          for i in range(28):
            x = Fake(shape=(1,2048,2048),dtype=dtypes.bfloat16)
            model.text_layer(x,None,None,f'vlm.model.language_model.layers.{i}',model.cache[i],None,0)
            assert session.state == state and session.frame is None
            assert mlps[-1][0] is session.fixed[f'vlm.model.language_model.layers.{i}/2']
            assert mlps[-1][2] is session.fixed[f'vlm.model.language_model.layers.{i}/3']
          assert len(mlps) == 28 and len(quants) == 84 and h.info['bound_program_calls'] == 0
          image_state = inspect.getclosurevars(session.original['quantize'].__func__).nonlocals['state']
          for images in (1,2):
            image_state['images'] = images
            for i in range(2,27):
              for fc,width in (('fc1',1152),('fc2',4304)):
                model.linear(Fake(shape=(images,720,width),dtype=dtypes.bfloat16),f'vlm.model.visual.encoder.layers.{i}.mlp.{fc}')
            image_state['images'] = 0
          assert len(visions) == 100 and len(mlps) == 28 and session.state == state
          h.original_matmul = lambda *a,**kw:h.prepared['current']['ast']
          def attention(q,k,v,mask):
            assert q.shape == (1,16,64,128) and k.shape == v.shape == (1,8,2368,128)
            assert session.frame is None and session.state['prefill'] is False and session.state['index'] == 0
            expert._matmul(*buffers,shape=(1,16,8,64,2368,128),scaled=True)
            selected = expert._matmul(*buffers,shape=P.SHAPE)
            assert selected is h.prepared['candidate']['program'] if candidate else selected is h.prepared['current']['ast']
            return q
          mocks.enter_context(patch.object(expert,'attention',attention))
          def trace(*args):
            for step in range(3):
              for i in range(28): model.text_layer(Fake(shape=(1,64,512),dtype=dtypes.bfloat16),None,None,
                f'expert.expert.layers.{i}',model.cache[i],Fake(shape=(2368,),dtype=dtypes.bfloat16),None)
          h.originals['_trajectory'] = trace
          before_q = len([e for e in model.events if e[0] == 'dynamic-quantize'])
          model._trajectory(None,None,None,2033,3,6)
          assert len([e for e in model.events if e[0] == 'dynamic-quantize'])-before_q == 336
          assert h.info['traces'] == [{'pv':84,'qk':84}] and len(mlps) == 28 and len(visions) == 100
          assert session.state == state and session.frame is None and h.depth == 0
          if candidate:
            program = h.prepared['candidate']['program']; assert program.op is Ops.PROGRAM
            params = [UOp.param(i,x.dtype,x.numel(),device='AMD') for i,x in enumerate(buffers)]
            assert _get_call_to_compile(program.call(*params)) is None
          h.depth = 1
          for shape,causal,scaled in ((P.SHAPE,True,False),(P.SHAPE,False,True),((2,16,8,64,128,2368),False,False),
              ((1,16,8,32,128,2368),False,False),((1,16,8,64,128,2112),False,False)):
            assert expert._matmul(*buffers,shape=shape,causal=causal,scaled=scaled) is h.prepared['current']['ast']
          h.depth = 0
          assert expert._matmul(*buffers,shape=P.SHAPE) is h.prepared['current']['ast']
          h.original_matmul,h.originals = actual_matmul,original_trajectory
          stream = C.load(C.MLP/'experiments/stream-inputs.py','expert_combined_cpu_stream')
          cmd = C.read(C.BASE.parent/'native-pipeline.json')['command']
          frames = stream.NativeFrames(*(C.option(cmd,k) for k in ('--native-manifest','--native-calibrations','--native-frame-cache')))
          row = C.read(C.option(cmd,'--native-manifest'))['schedule'][0]
          import numpy as np
          image,_ = frames.prepare(row['sample'])
          np.testing.assert_array_equal(image,np.load(Path(row['sample'])/'image_frames.npy',allow_pickle=False))
          assert comp.sampler_calls == 10 and comp.sampler.lib.provenance['library'].startswith(str(C.WORK))
          # Synthetic capture metadata tests coverage gates, not GPU arithmetic.
          comp.names = lambda jit:['agent_packed_epilogue_cpu']*28 if jit is model.prefill_jit else (
            ['agent_vision_tiles_3x3_b2']*25+['agent_vision_tiles_3x6_b2']*25)
          fakejit = type(model.prefill_jit)(); cache.encoders[((1,720,768),(1,720,1152))*2] = fakejit
          comp.vision_info['trace_calls'] = {k:v for k,v in comp.vision_info['trace_calls'].items() if k.endswith('/batch2')}
          proof = comp.coverage(); assert proof['all_28_mlp_and_50_vision_projections']
          missing = comp.vision_info['trace_calls'].pop('vlm.model.visual.encoder.layers.26.mlp.fc2/batch2')
          reject('missing projection/'+str(candidate),comp.coverage)
          comp.vision_info['trace_calls']['vlm.model.visual.encoder.layers.26.mlp.fc2/batch2'] = missing
          session.frame = object(); reject('active quantizer owner/'+str(candidate),comp.coverage); session.frame = None
          model.prefill_jit.cnt,model.prefill_jit.captured = 2,object()
          cache.cache['control'] = object()
          raise KeyboardInterrupt('Deliberate interrupted reverse cleanup')
      except KeyboardInterrupt: pass
      assert importlib.util.spec_from_file_location is original_spec and hcq2._build_queues is original_queues
      assert session.stats['methods_restored'] and h.info['cleanup_exact'] and h.info['observer_cleanup_exact']
      assert expert._matmul is actual_matmul and model.trajectory == original_trajectory['trajectory']
      assert not hasattr(model,'agent_expert_pv_qualified') and not hasattr(model,'agent_vision_tiles')
      assert not cache.cache and not cache.encoders and fakejit.resets == 1 and model.prefill_jit.resets == 1
      assert comp.sampler.pool is None and all(r['restored'] for r in comp.restorations)
      arms.append({'candidate':candidate,'language_layers':28,'vision_cpu_dispatches':100,'expert_pv_qk_each':84,
        'expert_quantizer_calls':336,'quantizer_owner_slots_restored':True,'sampler_snapshot_exact':True,
        'global_compiler_parallel_cache_unchanged':True,'interrupted_reverse_cleanup_exact':True,
        'complete_program_bypasses_worker':candidate,'events':comp.events})
    # Each real arm is a fresh process. Remove only test-created factory/input imports.
    for name in ('model','benchmark','inputs','native_warp','stream_inputs'): sys.modules.pop(name,None)
  R = C.load(HERE/'agent-expert-combined-run.py','expert_combined_cpu_runner')
  # These are synthetic comparator controls only. No four-query GPU equality is claimed.
  path = C.WORK/'agent-exact-combined-quality-candidate/report.json'
  baseline = C.read(path); baseline.pop('agent_exact_combined')
  rows = baseline['warmup']+baseline['runs']; selected = [rows[i] for i in C.INDICES]
  baseline['warmup'],baseline['runs'] = selected[:2],selected[2:]
  side = C.read(path.with_suffix('.speculation.json')); side['records'] = [side['records'][i] for i in C.INDICES]
  side['queries'] = 4; side['accepted'] = sum(r['accepted'] for r in side['records'])
  plan = {'warmup':2,'runs':2,'scheduled_samples':[r['sample'] for r in selected]}
  assert R.compare(C,baseline,baseline,plan,side,side)['samples'] == 4
  def leaf(x,value=None):
    while isinstance(x[0],list): x=x[0]
    x[0] = x[0]+.000001 if value is None else value
  for field in ('xyz','rot','tokens','text','pixels','cache','history','sixth_target','spec','signedzero','nonfinite','unknown','queryorder','count'):
    before,after,changed = copy.deepcopy(baseline),copy.deepcopy(baseline),copy.deepcopy(side)
    row = after['runs'][0]
    if field in ('xyz','rot'): leaf(row['prediction'][field])
    elif field == 'tokens': row['prediction']['generated_token_ids'][0] += 1
    elif field == 'text': row['prediction']['reasoning'] = 'mutated'
    elif field == 'pixels': row['uploaded_pixel_bytes'] += 1
    elif field == 'cache': row['vision_hits'] += 1
    elif field == 'history': after['history_pixel_scale'] = .5
    elif field == 'sixth_target': changed['records'][0]['targets'][5] += 1
    elif field == 'spec': changed['records'][0]['all_targets_finite'] = False
    elif field == 'signedzero': before['runs'][0]['control'] = 0.; row['control'] = -0.
    elif field == 'nonfinite': leaf(row['prediction']['xyz'],float('nan'))
    elif field == 'unknown': after['unreviewed_new_field'] = True
    elif field == 'queryorder': after['runs'].reverse()
    else: after['runs'].pop()
    reject('canonical/'+field,lambda:R.compare(C,before,after,plan,side,changed))
  compiler = C.paths().WGPCompiler
  code = compiler.__init__.__code__
  try:
    compiler.__init__.__code__ = (lambda self,arch:None).__code__
    reject('compiler runtime code mutation',C.paths)
  finally: compiler.__init__.__code__ = code
  with patch.object(sys.modules['wgp_compiler'],'__file__','/unrelated/wgp_compiler.py'): reject('foreign compiler import path',C.paths)
  Q = C.load(HERE/'agent-expert-combined-queue.py','expert_combined_cpu_queue')
  sequence = Q.sequence_checks()
  # Source-linked actual import path, including runpy's missing script-directory behavior.
  probe = R.import_probe(C)
  assert not Device._opened_devices
  C.write(args.output,{'complete':True,'gpu_executed':False,'arms':arms,'rejected':rejected,'import_probe':probe,
    'sequence':sequence,'scope':'CPU mocks prove routing and cleanup; parent four-query GPU composition remains unqualified',
    'sources':{str(p):C.sha(p) for p in HERE.glob('agent-expert-combined-*.py')},'prior':C.prior()},new=True)
  print({'complete':True,'gpu_executed':False,'rejections':len(rejected)})


if __name__ == '__main__': main()
