"""Bounded past-completion drafts. Numerical candidate, not serial equivalence."""
import ast
import copy
import inspect
import textwrap
import numpy as np

WIDTH = 16
NOMINAL = (124549,11135,3174,4286,1561,135084)


def complete(tokens, anchor):
  values = tuple(int(x) for x in tokens)
  if not values or values[-1] != anchor or anchor in values[:-1]:
    raise ValueError('Only a completed output with one final anchor may enter draft history')
  return values


def choose(previous, prompt, padded, max_tokens, anchor):
  if padded != 2048 or not 0 < prompt <= padded: raise ValueError('Require the original padded 2048-row prompt')
  if previous is None or tuple(previous) == NOMINAL or padded-prompt < WIDTH or max_tokens < WIDTH:
    return ()
  return complete(previous,anchor)[:WIDTH]


def prefix(payload, draft, anchor):
  values = np.asarray(payload)
  if values.shape != (2*(WIDTH+1),) or not np.issubdtype(values.dtype,np.integer):
    raise ValueError('Require all 17 target IDs and all 17 finite flags')
  if not 0 < len(draft) <= WIDTH or anchor in draft[:-1]: raise ValueError('Malformed draft prefix')
  targets, checks = values[:WIDTH+1], values[WIDTH+1:]
  if not np.isin(checks,(0,1)).all(): raise ValueError('Malformed finiteness flags')
  if not np.all(checks == 1): return 0, False
  accepted = 0
  for proposed, target in zip(draft,targets):
    if proposed != int(target): break
    accepted += 1
    if proposed == anchor: break
  return accepted, True


def contract(prompt, accepted, generated, capacity, max_tokens):
  if not 0 <= accepted <= generated <= max_tokens or prompt+generated > capacity:
    raise ValueError('Invalid generated/cache extent')
  return {'target_rows':list(range(prompt-1,prompt+WIDTH)),
    'retained_draft_rows':[prompt,prompt+accepted],
    'continuation_logit_row':prompt-1+accepted,
    'next_cache_write':prompt+accepted,'next_decode_rope_step':accepted,
    'final_cache_length':prompt+generated,'action_rotary_offset':generated}


def resume_lookup(original):
  """Two explicit AST edits; every original lookup/serial loop statement remains."""
  source = textwrap.dedent(inspect.getsource(original))
  tree = ast.parse(source); function = tree.body[0]
  assert isinstance(function,ast.FunctionDef) and function.name == 'generate'
  assert not function.decorator_list
  generated, warm = [], []
  for node in ast.walk(function):
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='generated':
      if isinstance(node.value,ast.List) and not node.value.elts: generated.append(node)
    if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(ast.parse('self.history',mode='eval').body):
      if any(isinstance(n,ast.Attribute) and n.attr=='warm_verifiers' for n in ast.walk(node)): warm.append(node)
  assert len(generated)==len(warm)==1, 'Original lookup source changed'
  old = copy.deepcopy(tree)
  generated[0].value = ast.parse('list(self._adaptive_seed)',mode='eval').body
  warm[0].test = ast.parse('self.history and not generated',mode='eval').body
  ast.fix_missing_locations(tree)
  # Reversibility rejects accidental changes to the original decoder body.
  inverse = copy.deepcopy(tree)
  inverse.body[0].body[0].value = ast.List(elts=[],ctx=ast.Load())
  for node in ast.walk(inverse):
    if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(warm[0].test):
      node.test = ast.parse('self.history',mode='eval').body
  assert ast.dump(inverse,include_attributes=False)==ast.dump(old,include_attributes=False)
  namespace = dict(original.__globals__)
  exec(compile(tree,inspect.getfile(original)+':adaptive-resume','exec'),namespace)
  return namespace['generate'], {'edits':['seed generated with the verified prefix',
    'do not warm verifier graphs over retained KV'], 'source_reversible':True}
