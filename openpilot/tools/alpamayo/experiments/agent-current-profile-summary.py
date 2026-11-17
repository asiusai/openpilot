"""Sum complete stage-local GPU event tables. Profiled runs cannot establish throughput."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
CATEGORIES = {'vision': 'vision', 'prefill': 'prefill', 'reasoning': 'decoder', 'diffusion': 'expert'}


def family(name):
  clean = ANSI.sub('', name).strip()
  if clean.startswith('copy '): return 'transfer'
  if clean.startswith('alpamayo_'):
    # Preserve specialized kernel names. Only remove dimension suffixes.
    return re.sub(r'(?:_\d+)+$', '', clean)
  if re.match(r'^[ErR]_\d', clean): return 'generated_' + clean.split('_')[0]
  return clean or '<unnamed>'


def aggregate(entries):
  if not entries: raise ValueError('No profiled queries')
  names = defaultdict(lambda: [0, [], 0., set()])
  stages_seen, invalid = set(), []
  for query, entry in enumerate(entries):
    profile = entry['device_profile']
    invalid.extend(dict(query=query, **item) for item in profile['invalid_samples'])
    for stage, value in profile['stages'].items():
      stages_seen.add(stage)
      totals = []
      for row in value['kernels']:
        count, total, maximum = row['count'], row['total_ms'], row['max_ms']
        if type(count) is not int or count <= 0: raise ValueError('Invalid event count')
        if not all(math.isfinite(x) and x >= 0 for x in (total, maximum)):
          raise ValueError('Invalid event duration')
        if maximum > total + 1e-8: raise ValueError('Maximum exceeds total')
        clean = ANSI.sub('', row['name']).strip()
        result = names[stage, clean]
        result[0] += count
        result[1].append(total)
        result[2] = max(result[2], maximum)
        result[3].add(row['name'])
        totals.append(total)
      actual = math.fsum(totals)
      if not math.isfinite(value['sum_kernel_ms']) or not math.isclose(actual, value['sum_kernel_ms'], abs_tol=1e-7, rel_tol=1e-10):
        raise ValueError('Stage total does not match complete kernel table')
  stages = {}
  for stage in sorted(stages_seen):
    kernels = [dict(name=name, raw_names=sorted(v[3]), family=family(name), count=v[0],
                    total_ms=math.fsum(v[1]), max_event_ms=v[2])
               for (s, name), v in names.items() if s == stage]
    kernels.sort(key=lambda row: (-row['total_ms'], row['name']))
    families = defaultdict(lambda: [0, []])
    for row in kernels:
      families[row['family']][0] += row['count']
      families[row['family']][1].append(row['total_ms'])
    family_rows = sorted([dict(family=k, count=v[0], total_ms=math.fsum(v[1]),
                              mean_ms_per_query=math.fsum(v[1])/len(entries)) for k, v in families.items()],
                         key=lambda row: (-row['total_ms'], row['family']))
    total = math.fsum(row['total_ms'] for row in kernels)
    transfers = math.fsum(row['total_ms'] for row in kernels if row['family'] == 'transfer')
    stages[stage] = dict(category=CATEGORIES.get(stage, stage), sum_gpu_events_ms=total,
                         mean_gpu_events_ms_per_query=total/len(entries), transfer_events_ms=transfers,
                         nontransfer_events_ms=total-transfers,
                         count=sum(row['count'] for row in kernels), families=family_rows, kernels=kernels)
  categories = {}
  for category in dict.fromkeys([*CATEGORIES.values(), *(v['category'] for v in stages.values())]):
    matching = [k for k, v in stages.items() if v['category'] == category]
    total = math.fsum(stages[k]['sum_gpu_events_ms'] for k in matching)
    categories[category] = dict(observed=bool(matching), stages=matching, sum_gpu_events_ms=total,
                                mean_gpu_events_ms_per_query=total/len(entries))
  all_families = defaultdict(lambda: defaultdict(list))
  for stage, values in stages.items():
    for row in values['families']: all_families[row['family']][stage].append(row['total_ms'])
  global_families = [dict(family=name, total_ms=math.fsum(x for values in by_stage.values() for x in values),
                          mean_ms_per_query=math.fsum(x for values in by_stage.values() for x in values)/len(entries),
                          stage_totals_ms={stage: math.fsum(values) for stage, values in by_stage.items()})
                     for name, by_stage in all_families.items()]
  global_families.sort(key=lambda row: (-row['total_ms'], row['family']))
  return dict(schema=1, diagnostic_only=True, timings_valid_for_throughput=False,
              queries=len(entries), samples=[entry['sample'] for entry in entries],
              valid_timestamp_coverage=not invalid, invalid_samples=invalid,
              sum_gpu_events_ms=math.fsum(v['sum_gpu_events_ms'] for v in stages.values()),
              categories=categories, stages=stages, families=global_families,
              limits=['GPU event sums are work durations, not critical-path latency or throughput.',
                      'Timestamp readbacks and synchronized stage fences perturb execution.',
                      'Transfer events remain in full sums and have separate subtotals.',
                      'An absent stage means no recorded events; it does not prove no work.',
                      'Prefill includes complete draft target verification; reasoning maps to decoder.'])


def summarize(report):
  return dict(measured=aggregate(report['runs']),
              warmup=aggregate(report['warmup']) if report.get('warmup') else None)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.output.write_text(json.dumps(summarize(json.loads(args.report.read_text())), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
