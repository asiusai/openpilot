"""Passive local/package/data inspection and public license/source retrieval. No GPU imports."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
EXP=Path(__file__).resolve().parent
ARCHIVE=Path('/mnt/nvme-256/footage/f9ad9a96')
PROTECTED='0000008e--321b2ddc03'
REV='33f9bf447ed3bcb7d545ce13f4226f824214fafb'
RECIPE='622f9d23ff9e4961ba8c6aac71367776cdf6b016'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def get(url,limit=32*1024*1024):
  request=urllib.request.Request(url,headers={'User-Agent':'Asius-read-only-feasibility-audit'})
  with urllib.request.urlopen(request,timeout=60) as response:
    content=response.read(limit+1)
    assert len(content)<=limit
    return content


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifacts',type=Path,default=WORK/'agent-distill-feasibility-artifacts')
  parser.add_argument('--public-docs',action='store_true')
  parser.add_argument('--log-channels',action='store_true')
  args=parser.parse_args();root=args.artifacts.resolve()
  assert root.is_relative_to(WORK) and root.name.startswith('agent-distill-feasibility-')
  root.mkdir(parents=True,exist_ok=True)
  result=dict(schema=1,gpu_initialized=False,training_executed=False,protected_route=PROTECTED)
  result['lspci']=subprocess.check_output(['lspci','-Dnn'],text=True)
  cards=[]
  for path in sorted(Path('/sys/bus/pci/devices').iterdir()):
    try:
      if (path/'vendor').read_text().strip()!='0x1002' or not (path/'class').read_text().strip().startswith('0x03'):continue
      row=dict(bdf=path.name,device=(path/'device').read_text().strip(),driver=str((path/'driver').resolve()))
      for field in ('mem_info_vram_total','mem_info_vis_vram_total'):
        if (path/field).exists():row[field]=int((path/field).read_text())
      cards.append(row)
    except (FileNotFoundError,PermissionError):pass
  result['passive_pci_cards']=cards
  topology=Path('/sys/class/kfd/kfd/topology/nodes')
  result['passive_kfd_properties']={path.name:(path/'properties').read_text() for path in topology.iterdir() if (path/'properties').exists()}
  packages={}
  for name in ('torch','torchvision','transformers','accelerate','safetensors','physical-ai-av','deepspeed','triton','flash-attn','pycapnp','zstandard'):
    try:
      dist=metadata.distribution(name);packages[name]=dict(version=dist.version,location=str(dist.locate_file('')))
      if name=='torch':packages[name]['version_source']=(Path(dist.locate_file('torch/version.py'))).read_text()
    except metadata.PackageNotFoundError:packages[name]=dict(installed=False)
  result['installed_packages']=packages
  summaries=[]
  for path in sorted((ARCHIVE/'realdata/.summaries').glob('*.json')):
    data=json.loads(path.read_text());route=data['routeId']
    segments=sorted((ARCHIVE/'realdata').glob(route+'--*'),key=lambda p:int(p.name.rsplit('--',1)[1]))
    complete=[p for p in segments if all((p/name).is_file() for name in ('fcamera.mp4','ecamera.mp4','rlog.zst'))]
    row=dict(route=route,protected=route==PROTECTED,start_unix_ms=data.get('startTime'),time_estimated=data.get('timeEstimated'),
      utc=datetime.fromtimestamp(data['startTime']/1000,timezone.utc).isoformat(),duration_s=data.get('durationSeconds'),
      distance_m=data.get('distanceMeters'),segments=len(segments),two_front_and_rlog_segments=len(complete),
      summary_sha256=sha(path),coords_count=len(data.get('coords',[])),
      drive_identity_limit='Logger route identity is present; cross-route physical-drive grouping is not established by this summary.')
    if args.log_channels and route!=PROTECTED and complete:
      sys.path[:0]=['/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
      from openpilot.cereal import log
      import zstandard
      # One bounded first available segment per route. Report coverage; never infer absence in uninspected segments.
      segment=complete[0];source=segment/'qlog.zst'
      if source.exists():
        with source.open('rb') as stream,zstandard.ZstdDecompressor().stream_reader(stream) as decoder:raw=decoder.read(32*1024*1024+1)
        assert len(raw)<=32*1024*1024
        counts=Counter();valid=Counter();nav=[];wall=[]
        for event in log.Event.read_multiple_bytes(raw):
          kind=event.which();counts[kind]+=1
          if event.valid:valid[kind]+=1
          if kind=='initData':wall.append(dict(monotonic_ns=int(event.logMonoTime),wall_time_ns=int(event.initData.wallTimeNanos)))
          if kind=='navInstruction':nav.append(dict(log_ns=int(event.logMonoTime),valid=bool(event.valid)))
        row['inspected_qlog']=dict(path=str(source),sha256=sha(source),uncompressed_bytes=len(raw),
          channels=dict(counts),valid_channels=dict(valid),navigation_records=nav,initialization=wall)
    summaries.append(row)
  result['owned_route_summaries']=summaries
  result['owned_inventory_transfer_status']=json.loads((ARCHIVE/'manifests/2026-11-16-office/dragon-status.json').read_text())
  nav_files=sorted((ARCHIVE/'exports').glob('**/*nav*.json'))
  result['existing_owned_navigation_manifests']=[dict(path=str(p),sha256=sha(p),protected_route=PROTECTED in str(p)) for p in nav_files]
  samples=[]
  for parent in (Path('/mnt/nvme-500/tmp'),Path('/tmp/opencode')):
    for path in sorted(parent.glob('*nvidia*/sample-*/metadata.json')):
      value=json.loads(path.read_text());prov=value.get('provenance',{})
      samples.append(dict(path=str(path),sha256=sha(path),clip_id=value.get('clip_id',prov.get('clip_id')),
        dataset_revision=prov.get('dataset_revision'),timestamp_us=value.get('t0_us'),nav_text=value.get('nav_text'),
        drive_id=prov.get('drive_id',value.get('drive_id')),shape=value.get('shape')))
  result['existing_nvidia_samples']=samples
  result['existing_nvidia_unique_clips']=sorted({sample['clip_id'] for sample in samples if sample['clip_id']})
  result['cached_dataset_revisions']=[p.name for p in Path('/home/john/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots').iterdir()]
  if args.public_docs:
    url=f'https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/revision/{REV}'
    data=json.loads(get(url));assert data['sha']==REV
    document=dict(url=url,sha=data['sha'],gated=data.get('gated'),cardData=data['cardData'],
      license_files=[r for r in data.get('siblings',[]) if 'license' in r['rfilename'].lower()],
      limit='Public gating terms returned for the pinned revision; gated LICENSE.pdf is separately tested without authentication.')
    target=root/'agent-distill-feasibility-dataset-terms.json';write(target,document)
    result['dataset_terms']=dict(path=str(target),sha256=sha(target))
    pdf=f'https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/{REV}/LICENSE.pdf'
    try:result['gated_license_pdf']=dict(url=pdf,status=200,bytes=len(get(pdf)))
    except urllib.error.HTTPError as error:result['gated_license_pdf']=dict(url=pdf,status=error.code)
    for filename,relative in (('sft.py','models/sft.py'),('SFT-README.md','README.md'),('sft-base.yaml','configs/sft_base.yaml')):
      url=f'https://raw.githubusercontent.com/NVlabs/alpamayo-recipes/{RECIPE}/recipes/alpamayo2edge_sft/{relative}'
      content=get(url,512*1024);target=root/('agent-distill-feasibility-'+filename);target.write_bytes(content)
      result.setdefault('public_recipe_sources',[]).append(dict(url=url,path=str(target),sha256=sha(target)))
  source_paths=[Path(__file__),BASE/'model/config.json',BASE/'model/model.safetensors.index.json',BASE/'model/LICENSE',
    BASE/'upstream/src/alpamayo2_edge/models/alpamayo2_edge.py',BASE/'tinygrad/tinygrad/mixin/gradient.py',
    BASE/'upstream/.venv/lib/python3.12/site-packages/transformers/models/cosmos3_edge/modeling_cosmos3_edge.py']
  result['source_sha256']={str(p):sha(p) for p in source_paths}
  assert not any(name=='torch' or name.startswith('tinygrad.runtime.ops_') for name in sys.modules)
  target=root/'agent-distill-feasibility-inventory.json';write(target,result)
  print(json.dumps(dict(output=str(target),owned_routes=len(summaries),nonprotected_two_front_routes=sum(not r['protected'] and r['two_front_and_rlog_segments']>0 for r in summaries),
    inspected_qlogs=sum('inspected_qlog' in r for r in summaries),nvidia_samples=len(samples),nvidia_unique_clips=result['existing_nvidia_unique_clips'],
    internal_cards=cards,gpu_initialized=False),indent=2))


if __name__=='__main__':main()
