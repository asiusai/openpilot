"""Read ELF kernel metadata without loading or executing vendor code."""
import argparse
import json
import struct
from pathlib import Path
import msgpack

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
data=args.source.read_bytes()
assert data[:6]==b'\x7fELF\x02\x01'
offset=struct.unpack_from('<Q',data,40)[0]
size,count,names=struct.unpack_from('<HHH',data,58)
assert size==64
sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
strings=sections[names]
strings=data[strings[4]:strings[4]+strings[5]]
result=[]
for section in sections:
  if section[1]!=7: continue
  start,end=section[4],section[4]+section[5]
  while start<end:
    name_size,desc_size,note_type=struct.unpack_from('<III',data,start)
    start+=12
    name=data[start:start+name_size].rstrip(b'\0').decode()
    start+=(name_size+3)//4*4
    descriptor=data[start:start+desc_size]
    start+=(desc_size+3)//4*4
    if name=='AMDGPU' and note_type==32:
      result.append(msgpack.unpackb(descriptor,raw=False,strict_map_key=False))
assert result
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('metadata records',len(result),'kernels',sum(len(record.get('amdhsa.kernels',[])) for record in result))
