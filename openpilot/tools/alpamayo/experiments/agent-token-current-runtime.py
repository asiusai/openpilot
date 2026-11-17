"""Adapt the frozen pruning controller through its selector interface only.

Never copy model arithmetic or alter shared/core source files. Each factory owns
an isolated imported controller and reference facade, including one fixed policy.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent


def module(name, namespace):
  spec = importlib.util.spec_from_file_location(namespace, HERE/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[namespace] = value
  spec.loader.exec_module(value)
  return value


policy_module = module('agent-token-current-policy', 'agent_token_current_runtime_policy')


def make_runtime(policy):
  if policy not in policy_module.POLICIES: raise ValueError('Unsupported retention policy')
  base = policy_module.load_frozen('agent-token-prune-runtime', 'agent_token_current_frozen_runtime_'+uuid.uuid4().hex)
  # This private imported module's globals are isolated from the original route.
  # The original source and any previously imported original selector stay intact.
  original_reference = base.reference
  base.reference = SimpleNamespace(full_plan=original_reference.full_plan,
    select_last_query=lambda scores, prompt, capacity, frames:policy_module.select(scores, prompt, capacity, frames, policy))

  class CurrentPruner(base.NumericalPruner):
    def prepare(self, prompt, frames):
      self.metadata = None  # Invalid observations cannot reuse a previous prediction.
      frames, _, _ = policy_module.validate_bound_frames(int(prompt), frames, require_full_resolution=self.require_packed)
      return super().prepare(prompt, frames)

    def padded_prefill(self, x, cos, sin, max_tokens, length):
      result = super().padded_prefill(x, cos, sin, max_tokens, length)
      event = self.events[-1]
      # Replace the inherited uniform50% event label; arithmetic is inherited unchanged.
      event.pop('retention', None)
      event.update(classification='NUMERICAL NEW MODEL; latest full, whole-current-prompt-conditioned; not exact',
                   policy=policy, latest_image_retention=1.0, older_image_retention=policy_module.POLICIES[policy],
                   frame_retention=policy_module.frame_counts(self.last_plan),
                   selection_source='original layer1 last-real-prompt-query attention; no generated tokens or route outputs',
                   generation_target='changed numerical model', prompt_prefix_invariance_pass=False,
                   quality_passed=None)
      return result

  return SimpleNamespace(NumericalPruner=CurrentPruner,
                         current_observations=policy_module.current_observations,
                         prepared_observations=policy_module.prepared_observations,
                         frame_spans=policy_module.frame_spans, reference=original_reference,
                         policy=policy, controller_source_sha256=hashlib.sha256((HERE/'agent-token-prune-runtime.py').read_bytes()).hexdigest())
