"""Freeze existing dependencies into the explicitly owned trial directory, CPU only."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE/'agent-streaming-plan-parent-trial'

def main():
  if ROOT.exists():raise FileExistsError(ROOT)
  target=ROOT/'alpamayo';experiment=target/'experiments'
  experiment.mkdir(parents=True)
  sources=list(HERE.parent.glob('*.py'))
  sources += [p for p in HERE.iterdir() if p.is_file() and
              (p.suffix in ('.py','.c') and not p.name.startswith('agent-') or p.name=='coc_vocab.json')]
  sources += [HERE/'agent-attention-rowsoftmax.py']
  records=[]
  for source in sorted(sources):
    content=source.read_bytes()
    destination=(experiment if source.parent==HERE else target)/source.name
    destination.write_bytes(content)
    digest=hashlib.sha256(content).hexdigest()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    records.append(dict(source=str(source),copy=str(destination.relative_to(ROOT)),sha256=digest))
  scales=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/prefill-calibration/report.prefill-scales.json')
  (ROOT/'fixed-language-scales.json').write_bytes(scales.read_bytes())
  records.append(dict(source=str(scales),copy='fixed-language-scales.json',sha256=hashlib.sha256(scales.read_bytes()).hexdigest()))
  invocation=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/token-current-source/agent-token-current-quality-invocation.json')
  saved=json.loads(invocation.read_text())
  # Explicit correctness configuration: serial reasoning in both paired runs.
  # Preserve selected math; no fixed vision scales, pruning, or input reductions.
  args=['--checkpoint','/tmp/opencode/alpamayo-asius-20261114/model','--samples','unused',
    '--output','unused','--initial-noise','/tmp/opencode/alpamayo-nvidia-reference-full/initial_noise.npy',
    '--reduced-vocab',str(experiment/'coc_vocab.json'),'--camera-profile','nvidia','--allow-missing-cameras',
    '--mmap-inputs','--expert-kernels','--compact-inputs','--byte-pixels','--selective-pixels',
    '--cache-resize','--materialize-quant','--quantize-reduction','--grouped-quantize-reduction',
    '--packed-quantize','--decode-block','1','--vision-cache','--vision-cache-capacity','48',
    '--group-vision','--vision-group-limit','2','--batched-vision','--packed-projector','--vision-views',
    '--online-attention','--fused-norm','--llvm-wgp','--vision-batch','1','--pixel-scale','1',
    '--steps','3','--warmup','0','--runs','32','--device-timeout-ms','10000',
    '--bf16-vision-layers','2','--bf16-expert-projections','--packed-bf16-vision',
    '--max-tokens','64','--pad-prompt-to','2048','--prefill-calibration',str(ROOT/'fixed-language-scales.json'),
    '--direct-byte-inputs','--cache-rotary-inputs']
  environment=dict(saved['environment'],PYTHONPATH=f'/tmp/opencode/alpamayo-asius-20261114/tinygrad:{target}',
                   CACHEDB=str(ROOT/'parent-gpu-cache.db'))
  config=dict(args=args,environment=environment,checkpoint_revision='b92b6aa326bd3e2235db28f248ad5a30b5890e64',
    prefill_attention='shuffle-fma',prefill_waves=8,reasoning='serial in both paired runs; correctness prototype',
    dynamic_vision=True,full_five_images_each=True,euler_steps=3)
  (ROOT/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
  paths=json.loads(Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-route-samples.json').read_text())[:32]
  (ROOT/'smoke32.json').write_text(json.dumps(paths,indent=2)+'\n')
  (ROOT/'sources.json').write_text(json.dumps(dict(records=records,configuration_sha256=hashlib.sha256(
    (ROOT/'configuration.json').read_bytes()).hexdigest(),smoke32_sha256=hashlib.sha256((ROOT/'smoke32.json').read_bytes()).hexdigest()),indent=2)+'\n')
  print(json.dumps(dict(root=str(ROOT),copied_files=len(records),gpu_opened=False)))

if __name__=='__main__':main()
