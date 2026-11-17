"""Parent-only entry point: unchanged offline parity or unchanged paced replay.

Use the immutable snapshot and its pinned dynamic normalization adapter.
The source camera remains 20 Hz. Requested 10 Hz is only a scheduling ceiling.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def policy_module():
  spec = importlib.util.spec_from_file_location('agent_paced_dynamic_policy', ROOT/'agent-paced-dynamic-policy.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, add_help=False)
  parser.add_argument('--task', choices=('replay', 'paced'), required=True)
  parser.add_argument('--mode', required=True)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--decoder')
  parser.add_argument('--prefill-attention')
  parser.add_argument('--consumer-parity-report', type=Path)
  parser.add_argument('--hz', type=float)
  parser.add_argument('--start-seconds', type=float)
  parser.add_argument('--seconds', type=float)
  parser.add_argument('--inference-budget-seconds', type=float)
  options = list(sys.argv[1:] if argv is None else argv)
  args, _ = parser.parse_known_args(options)
  if options.count('--task') != 1:
    raise ValueError('Specify one explicit task')
  index = options.index('--task')
  del options[index:index+2]
  os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  sys.dont_write_bytecode = True
  sys.path.insert(0, str(ROOT))
  policy = policy_module()
  policy.verify_snapshot()
  expected = ROOT.parent.parent/'pipeline-invocation.json'
  if args.invocation.resolve() != expected:
    raise ValueError('Use the prepared nested pipeline invocation')
  if args.task == 'replay':
    if args.mode not in ('reference', 'consumer') or args.decoder != 'speculative' or args.prefill_attention != 'shuffle-fma':
      raise ValueError('Both parity passes require speculative prefill and shuffle-FMA')
    name = 'agent-live-replay'
  else:
    if (args.mode != 'run' or args.hz != 10 or args.start_seconds != 35 or args.seconds != 8 or
        args.inference_budget_seconds != 180 or args.consumer_parity_report is None):
      raise ValueError('One recorded 35..43 second run, requested 10 Hz, hard 180 second budget, and matching parity are required')
    policy.qualified_gate(json.loads(args.consumer_parity_report.read_text()))
    name = 'agent-paced-replay'
  with policy.imports():
    policy.local(name).main(options)


if __name__ == '__main__':
  main()
