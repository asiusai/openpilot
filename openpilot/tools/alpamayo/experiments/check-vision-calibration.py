"""Check complete scale accumulation through two independent captured graphs."""
import importlib.util
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes

spec = importlib.util.spec_from_file_location('vision_calibration',Path(__file__).with_name('vision-calibration.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Model:
  def grouped_quantize(self,x,images):
    grouped=x.float().reshape(images,-1,x.shape[-1])
    scales=(grouped.abs().max(axis=(1,2),keepdim=True)/448).clamp(min_=1e-12).realize()
    quantized=(grouped/scales).clamp(-448,448).cast(dtypes.fp8e4m3).reshape(-1,x.shape[-1]).contiguous()
    return quantized,scales.reshape(images,1).expand(images,grouped.shape[1]).reshape(-1).contiguous()

  def vision_layer(self,x,index):
    values=[]
    for factor in (1.,2.,.5,4.):
      quantized,_=self.grouped_quantize(x*factor,x.shape[0])
      values.append(quantized.bitcast(dtypes.uint8).reshape(-1))
    return Tensor.cat(*values).realize()


def main():
  ordinary,captured=Model(),Model()
  module.install(captured,capture=True)
  graphs={images:TinyJit(lambda x:captured.vision_layer(x,2)) for images in (1,2)}
  rng=np.random.default_rng(9060)
  maximum=0.
  for index in range(12):
    images=1+index%2
    x=rng.normal(size=(images,5,16)).astype(np.float32)*(index+1)
    maximum=max(maximum,float(np.abs(x).max()))
    value=Tensor(x)
    np.testing.assert_array_equal(graphs[images](value).numpy(),ordinary.vision_layer(value,2).numpy())
  actual={key:float(value.numpy()[0]) for key,value in captured.vision_activation_scales.items()}
  expected={f'vlm.model.visual.encoder.layers.2/{index}':maximum*factor/448
            for index,factor in enumerate((1.,2.,.5,4.))}
  assert actual.keys()==expected.keys()
  for key in expected: np.testing.assert_allclose(actual[key],expected[key],rtol=1e-6)
  fixed=Model()
  module.install(fixed,fixed_scales=actual)
  for images in (1,2):
    x=rng.normal(size=(images,5,16)).astype(np.float32)*13
    output=fixed.vision_layer(Tensor(x),2).numpy()
    expected_bytes=[]
    for index,factor in enumerate((1.,2.,.5,4.)):
      scale=actual[f'vlm.model.visual.encoder.layers.2/{index}']
      expected_bytes.append(((Tensor(x)*factor)/scale).clamp(-448,448).cast(dtypes.fp8e4m3).bitcast(dtypes.uint8).numpy().reshape(-1))
    np.testing.assert_array_equal(output,np.concatenate(expected_bytes))
  print('PASS: every captured/replayed batch contributes; dynamic output bytes unchanged; fixed scalar broadcasts match')


if __name__=='__main__': main()
