"""Record read-only primary-source provenance and an illustrative prefill cost bound.

Fetches source text only. Never downloads binaries, loads checkpoints, or runs inference.
Saved timing reports supply timing fields only; they do not select tokens.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import urllib.request

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
FASTV_COMMIT = 'd1659729b5bf1be225e99ee15783deeea80f63b1'
FASTV_PATH = 'src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py'


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-prune-research-report.json')
  args = parser.parse_args()
  sources = []
  for path in ('README.md', FASTV_PATH):
    url = f'https://raw.githubusercontent.com/pkunlp-icler/FastV/{FASTV_COMMIT}/{path}'
    with urllib.request.urlopen(url, timeout=60) as response: data = response.read()
    text = data.decode()
    markers = ('Support KV Cache', 'static kv-cache pruning', 'no speed up',
               'last_layer_attention_avg_last_tok =', 'position_ids = keep_indexs', 'cache_position = cache_position[:new_seq_length]')
    locations = {marker: [i for i, line in enumerate(text.splitlines(), 1) if marker in line]
                 for marker in markers if marker in text}
    sources.append({'url': url, 'commit': FASTV_COMMIT, 'sha256': hashlib.sha256(data).hexdigest(),
                    'bytes': len(data), 'marker_lines': locations})
  local_paths = ('model.py', 'inputs.py', 'experiments/block-decode.py', 'experiments/draft-decode.py',
                 'experiments/lookup-decode.py', 'experiments/padded-prefill.py', 'experiments/quantize-packed.py',
                 'experiments/expert-kernels.py', 'experiments/compact-inputs.py', 'experiments/online-attention.py')
  local = {path: hashlib.sha256((MODEL/path).read_bytes()).hexdigest() for path in local_paths}
  fraction_linear = (2*2112+26*1152)/(28*2112)
  fraction_attention = (2*2112**2+26*1152**2)/(28*2112**2)
  timing = json.loads((WORK/'full-native-5hz-fixed-byte/report.json').read_text())
  rows = [r['prediction']['timings_ms'] for r in timing['runs']]
  report = {
    'observed_date': '2026-11-16', 'source_text_only': True, 'retention_predeclared': 0.5,
    'prune_after_full_language_layers': 2, 'sources': sources, 'local_source_sha256': local,
    'papers': [
      {'url': 'https://arxiv.org/abs/2403.06764v3', 'title': 'An Image is Worth 1/2 Tokens After Layer 2',
       'limit': 'General VLM benchmarks, not Alpamayo Edge trajectory quality or its action-expert KV interface.'},
      {'url': 'https://github.com/pkunlp-icler/FastV/issues/14',
       'limit': 'Authors distinguish last-query scoring and static cache pruning from per-output re-pruning.'},
      {'url': 'https://arxiv.org/abs/2507.23318v4', 'title': 'FastDriveVLA',
       'limit': 'Reports driving weaknesses of generic visual-attention pruning; its proposed pruner requires training.'},
      {'url': 'https://arxiv.org/abs/2610.22146v2', 'title': 'FastVLA',
       'limit': 'Task-to-vision ranking on manipulation policies. Does not validate Alpamayo or this fixed budget.'},
    ],
    'illustrative_shapes': {'original_real_prompt': 2027, 'original_padded_rows': 2112, 'image_tokens': 1800,
                            'nonimage_tokens': 227, 'retained_image_tokens': 900, 'retained_real_rows': 1127,
                            'retained_rows_padded_64': 1152, 'linear_flop_fraction': fraction_linear,
                            'attention_flop_fraction': fraction_attention,
                            'vision_encoder_and_projector_unchanged': True, 'fixed_capacity_expert_attention_unchanged': True},
    'older_synchronized_timing_evidence': {
      'path': str(WORK/'full-native-5hz-fixed-byte/report.json'),
      'sha256': hashlib.sha256((WORK/'full-native-5hz-fixed-byte/report.json').read_bytes()).hexdigest(),
      'runs': len(rows), 'median_total_ms': statistics.median(r['total'] for r in rows),
      'median_prefill_ms': statistics.median(r['prefill'] for r in rows),
      'median_total_minus_prefill_ms': statistics.median(r['total']-r['prefill'] for r in rows),
      'median_linear_cost_projection_ms': statistics.median(r['total']-r['prefill']*(1-fraction_linear) for r in rows),
      'projection_is_not_measurement': True,
      'limits': 'Only an optimistic cost illustration on the older synchronized baseline. No selector, transfer, scatter, graph, or fallback overhead included.'},
    'gpu_hook_ready': False,
  }
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'output': str(args.output), 'primary_source_files': len(sources),
                    'older_timing': report['older_synchronized_timing_evidence'],
                    'linear_flop_fraction': fraction_linear, 'attention_flop_fraction': fraction_attention}, indent=2))


if __name__ == '__main__': main()
