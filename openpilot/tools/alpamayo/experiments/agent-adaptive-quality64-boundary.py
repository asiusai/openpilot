"""Observe full query clocks; rewind logical history only between timing passes."""
import ast
import copy
import inspect
from pathlib import Path
import runpy
import textwrap
from unittest.mock import patch


class Boundary:
  def __init__(self,source,overlay,timing):
    self.source=Path(source);self.overlay=overlay;self.timing=timing
    self.records=[];self.active=None;self.cold=0;self.components=None;self.restored=False;self.proof=None

  def state(self):
    d=self.components.decoder
    previous=None if self.overlay is None else self.overlay.states[0].previous
    return {'history':copy.deepcopy(d.history),'adaptive_previous':None if previous is None else list(previous),
      'adaptive_completed':None if self.overlay is None else self.overlay.states[0].completed}

  def attach(self,components):
    assert self.components is None;self.components=components
    assert components.decoder.history==[] and not components.vision_cache.cache

  def begin(self,index,components):
    assert components is self.components and self.active is None
    if self.timing and index==64:
      # Retain compiled graphs and buffers, never previous images or output history.
      d=components.decoder;c=components.vision_cache
      d.history=[];d.statistics=[];d.prefill_draft_old_history=[]
      c.cache.clear();c.keys=();c.output_grids=None;c.hits=c.misses=0
      if self.overlay:
        s=self.overlay.states[0];assert not s.running and s.current is None and s.active is None
        assert not s.audit_prefix_once and not s.audit_done and s.completed==64
        s.previous=None;s.completed=0
      assert not d.history and not c.cache and not c.keys
    from tinygrad.engine import realize
    self.active={'index':index,'original64_index':index%64,'phase':'measured' if self.timing and index>=64 else 'quality_or_warm',
      'before':self.state(),'program_cache_before':len(realize.to_program_cache),'runtime_cache_before':len(realize.runtime_cache)}
    self.cold=0

  def end(self,index,components):
    from tinygrad.engine import realize
    assert self.active['index']==index and components is self.components
    value=self.active
    value.update(after=self.state(),program_cache_after=len(realize.to_program_cache),runtime_cache_after=len(realize.runtime_cache),cold_jit_calls=self.cold)
    assert value['program_cache_after']>=value['program_cache_before'] and value['runtime_cache_after']>=value['runtime_cache_before']
    value['exclude_from_paired_timing']=bool(self.cold or value['program_cache_after']!=value['program_cache_before'] or
      value['runtime_cache_after']!=value['runtime_cache_before'])
    self.records.append(value);self.active=None

  def install_main(self,namespace):
    original=namespace['main'];tree=ast.parse(textwrap.dedent(inspect.getsource(original)));before=copy.deepcopy(tree)
    calls={'components = initialize(args, sources=sources, source_dir=source_dir)':'_quality64_boundary.attach(components)',
      'iteration_started = time.perf_counter()':'_quality64_boundary.begin(index, components)',
      'done = time.perf_counter()':'_quality64_boundary.end(index, components)'}
    hits=[]
    class Insert(ast.NodeTransformer):
      def visit_Assign(self,node):
        text=ast.unparse(node)
        if text not in calls:return node
        hits.append(text);extra=ast.parse(calls[text]).body[0]
        return [extra,node] if text.startswith('iteration_started') else [node,extra]
    tree=Insert().visit(tree);assert sorted(hits)==sorted(calls)
    class Remove(ast.NodeTransformer):
      def visit_Expr(self,node):
        return None if ast.unparse(node) in calls.values() else node
    inverse=Remove().visit(copy.deepcopy(tree));assert ast.dump(inverse)==ast.dump(before)
    namespace['_quality64_boundary']=self
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+':quality64-boundaries','exec'),namespace)
    self.proof={'source_reversible':True,'edits':list(calls.values()),'numeric_statements_changed':False,
      'clock':'Original done timestamp remains before observer; begin/reset occurs before original iteration clock.'}

  def __enter__(self):
    from tinygrad.engine.jit import TinyJit
    previous=runpy.run_path;owner=self;old_call=TinyJit.__call__
    def call(jit,*a,**kw):
      if owner.active is not None and jit.cnt<2:owner.cold+=1
      return old_call(jit,*a,**kw)
    def run(path,*a,**kw):
      if Path(path).resolve()!=owner.source:return previous(path,*a,**kw)
      namespace=previous(path,run_name='quality64_original_pipeline')
      # runpy returns a copy; functions retain their actual execution globals.
      ns=namespace['main'].__globals__;owner.install_main(ns);ns['main']();return ns
    self.patch_run=patch.object(runpy,'run_path',run);self.patch_jit=patch.object(TinyJit,'__call__',call)
    self.patch_run.__enter__();self.patch_jit.__enter__();return self

  def __exit__(self,*error):
    self.patch_jit.__exit__(*error);self.patch_run.__exit__(*error);self.restored=True
