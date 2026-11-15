"""Compare native predictions with saved, unmodified NVIDIA reference outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--native', type=Path, required=True)
  parser.add_argument('--reference', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  native = json.loads(args.native.read_text())
  reference = json.loads((args.reference/'report.json').read_text())
  assert native['initial_noise_sha256'] == reference['noise_sha256'], 'Diffusion inputs differ'
  comparisons = []
  for row in native['warmup'] + native['runs']:
    sample = Path(row['sample']).name
    path = args.reference/sample/'output.safetensors'
    if not path.exists():
      continue
    reference_row = next(item for item in reference['samples'] if item['sample_id'] == sample)
    manifest_checked = 'input_sha256' in reference_row
    if manifest_checked:
      metadata = json.loads((Path(row['sample'])/'metadata.json').read_text())
      assert metadata['sha256'] == reference_row['input_sha256'], f'Frozen inputs differ: {sample}'
      nav = [str(np.load(Path(row['sample'])/'nav_text.npy',allow_pickle=False).item())]
      assert nav == reference_row['nav_text'], f'Navigation differs: {sample}'
    expected = load_file(path)
    actual = row['prediction']
    a = np.asarray(actual['xyz']).reshape(64,3)
    b = expected['pred_xyz'].reshape(64,3)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    actual_rot = np.asarray(actual['rot']).reshape(64,3,3)
    expected_rot = expected['pred_rot'].reshape(64,3,3)
    assert np.isfinite(actual_rot).all() and np.isfinite(expected_rot).all()
    rotation_delta = np.linalg.norm(actual_rot-expected_rot,axis=(1,2))
    delta = np.linalg.norm(a-b,axis=-1)
    ids = expected['generated_ids'].reshape(-1)
    ids = ids[ids >= 0].tolist()
    exact_tokens = ids == actual['generated_token_ids']
    saved = json.loads((path.parent/'output.json').read_text())
    comparisons.append(dict(sample=sample, tokens_identical=exact_tokens, input_manifest_checked=manifest_checked,
                            native_text=actual['reasoning'], reference_text=saved['text'],
                             mean_position_delta_m=float(delta.mean()), max_position_delta_m=float(delta.max()),
                             max_rotation_matrix_delta=float(rotation_delta.max()),
                            position_delta_1_3_6_4s_m=delta[[9,29,63]].tolist(),
                            native_endpoint=a[-1].tolist(), reference_endpoint=b[-1].tolist(),
                            reference_output_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
  assert comparisons, 'No matching reference samples'
  report = dict(native=str(args.native), reference=str(args.reference),
                native_pixel_scale=native['pixel_scale'], reference_pixel_scale=reference['pixel_scale'],
                 native_camera_pixel_scales=native.get('camera_pixel_scales'),
                 reference_camera_pixel_scales=reference.get('camera_pixel_scales'),
                native_steps=native['settings']['diffusion_steps'],
                reference_steps=reference['settings']['diffusion_kwargs']['inference_step'],
                noise_sha256=reference['noise_sha256'],
                 compared=len(comparisons), identical_token_sequences=sum(row['tokens_identical'] for row in comparisons),
                 native_predictions=len(native['warmup'])+len(native['runs']),
                 reference_predictions=len(reference['samples']),
                mean_position_delta_m=float(np.mean([row['mean_position_delta_m'] for row in comparisons])),
                max_position_delta_m=max(row['max_position_delta_m'] for row in comparisons), samples=comparisons)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({key:value for key,value in report.items() if key != 'samples'},indent=2))


if __name__ == '__main__':
  main()
