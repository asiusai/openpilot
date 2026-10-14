# /// script
# requires-python = ">=3.11"
# dependencies = ["edge-tts==7.3.1"]
# ///
"""Generate bundled speech: uv run openpilot/selfdrive/assets/sounds/voice/generate.py.

Only generation uses the network. The device plays the resulting WAV files offline.
Requires ffmpeg. Use --check to verify files without contacting the speech service.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import wave

ROOT = Path(__file__).resolve().parent
VOICE = "en-US-JennyNeural"
# The normal mixer gain is 8192 (unity). Never add the mixer maximum's 8x boost.
# Keep stereo during loudness measurement to match the accepted asius-1 bench sample.
FILTER = ",".join([
  "aresample=48000,aformat=channel_layouts=stereo,highpass=f=180",
  "loudnorm=I=-11:TP=-1:LRA=4,aresample=48000,highpass=f=280",
  "acompressor=threshold=0.05:ratio=3:attack=5:release=120:makeup=10",
  "alimiter=limit=0.85:attack=5:release=80:level=false",
])


def signature(messages):
  return hashlib.sha256(json.dumps([messages, VOICE, FILTER], sort_keys=True).encode()).hexdigest()


def check(messages):
  metadata = json.loads((ROOT / "generated.json").read_text())
  if metadata["source_hash"] != signature(messages):
    raise ValueError("Speech text or processing changed. Regenerate the clips.")
  if set(metadata["clips"]) != set(messages) or {p.stem for p in ROOT.glob("*.wav")} != set(messages):
    raise ValueError("Speech files do not match the catalog. Regenerate the clips.")
  for key in messages:
    path = ROOT / f"{key}.wav"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["clips"][key]["sha256"]:
      raise ValueError(f"Speech file changed: {key}")
    with wave.open(str(path)) as wav:
      if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 48000):
        raise ValueError(f"Invalid audio format: {key}")
      if not 0 < wav.getnframes() <= 8 * 48000:
        raise ValueError(f"Invalid audio duration: {key}")
  print(f"Verified {len(messages)} offline speech clips.")


async def generate(messages):
  import edge_tts

  semaphore = asyncio.Semaphore(3)
  previous = json.loads((ROOT / "generated.json").read_text())["clips"] if (ROOT / "generated.json").exists() else {}
  metadata = {"voice": VOICE, "source_hash": signature(messages), "filter": FILTER, "clips": {}}
  with tempfile.TemporaryDirectory(prefix="openpilot-voice-") as tmp:
    async def clip(key, item):
      async with semaphore:
        raw = Path(tmp) / f"{key}.mp3"
        for attempt in range(3):
          try:
            await edge_tts.Communicate(item["text"], VOICE).save(str(raw))
            break
          except Exception:
            if attempt == 2:
              raise
            await asyncio.sleep(2 * (attempt + 1))
        wav = Path(tmp) / f"{key}.wav"
        await asyncio.to_thread(subprocess.run, ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw), "-af", FILTER,
                                                "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav)) as audio:
          seconds = audio.getnframes() / audio.getframerate()
        if not 0 < seconds <= 8:
          raise ValueError(f"Speech must be shorter than eight seconds: {key}")
        metadata["clips"][key] = {"seconds": round(seconds, 3), "sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}
        wav.replace(ROOT / wav.name)
        print(f"{key}: {seconds:.2f}s", flush=True)

    await asyncio.gather(*(clip(key, item) for key, item in messages.items()))
  # Remove only obsolete clips recorded by the previous generation.
  for key in previous.keys() - messages.keys():
    (ROOT / f"{key}.wav").unlink(missing_ok=True)
  (ROOT / "generated.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  check(messages)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--directory", type=Path, help="Speech catalog directory, including messages.json")
  args = parser.parse_args()
  if args.directory:
    ROOT = args.directory.resolve()
  messages = json.loads((ROOT / "messages.json").read_text())
  if args.check:
    check(messages)
  else:
    asyncio.run(generate(messages))
