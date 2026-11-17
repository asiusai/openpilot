"""Align only the negative-control read window. Target writes remain typed physical views."""

def neighbors(buffer,index,dt):
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp
  assert dt in (dtypes.uint16,dtypes.uint32)
  size=dt.itemsize;base=buffer.base;offset=buffer.offset
  assert buffer.nbytes%size==0 and offset%size==0
  assert 0<=offset and 0<buffer.nbytes<=base.nbytes-offset
  assert 0<=index<buffer.nbytes//size
  address=int(buffer.get_storage().buf);base_address=int(base.get_storage().buf)
  assert base_address%4==0 and address==base_address+offset
  lo=offset+max(0,index-1)*size;hi=offset+min(buffer.nbytes//size,index+2)*size
  start=lo//4*4;end=(hi+3)//4*4
  assert 0<=start<end<=base.nbytes,'Aligned read would exceed base storage'
  view=base.view(end-start,start).ensure_allocated()
  assert view.base is base and view.offset==start and view.nbytes==end-start
  assert int(view.get_storage().buf)==base_address+start and int(view.get_storage().buf)%4==0
  relative=(offset+index*size-start)//size
  assert start+relative*size==offset+index*size and 0<=relative<(end-start)//size
  return Tensor(UOp.from_buffer(view,dt)),relative,{'base_offset_bytes':start,'bytes':end-start,
    'base_bytes':base.nbytes,'gpu_address':base_address+start,'target_word_index':relative,
    'extra_prefix_bytes':lo-start,'extra_suffix_bytes':end-hi}
