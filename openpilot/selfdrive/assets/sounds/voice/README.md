# Voice alerts

`soundd` plays these optional clips for screen alerts and offroad faults.
Enable `VoiceAlertsEnabled` through the Asius app's Controls page.
The persistent toggle defaults to off. Playback works without internet.

`messages.json` lists 13 phrases and their event or offroad-parameter bindings.
The event names come from `selfdrive/selfdrived/events.py`, which supplies the
comma four `mici` alert renderer. These are selected common conditions, not
frequency statistics from driving logs. Both wheel-touch warnings use
"Touch the steering wheel." This is not a driver identity check.

The `no_entry` bindings speak only after a refused engagement attempt.
Normal pedal disengagement and passive vehicle readiness conditions stay silent.
Routine resume, brake-hold and lane-change prompts have no voice clips.
Rare faults share a general message. The app shows the specific fault.
Known camera and driving-model process failures keep their specific phrases.

Generate from the repository root:

```sh
uv run openpilot/selfdrive/assets/sounds/voice/generate.py
python3 openpilot/selfdrive/assets/sounds/voice/generate.py --check
```

Generation needs ffmpeg and sends only the fixed phrases to Microsoft's speech
service through pinned `edge-tts`. The device needs neither dependency.
The voice is `en-US-JennyNeural`. `generated.json` records input and output hashes.
Generation removes obsolete clips listed in the previous generated manifest.
Commit the mono 48 kHz, 16-bit WAV files with their manifest. These small clips
use ordinary Git storage, without uploading to comma's LFS server.

Processing uses the louder asius-1 speech test settings: loudness normalization,
speech-band filtering, compression and a peak limiter. Keep the Dragon mixer
at 8192, its normal unity gain. The maximum mixer value adds almost 8x gain and
clips these files. This is a tested playback setting, not a calibrated SPL rating.

Speech reserves 15% of full scale for existing noncritical chimes. Critical
sirens interrupt speech and retain their original samples, ramp and escalation.
The callback only reads preloaded samples. It never synthesizes, decodes files,
reads parameters or starts a subprocess. The control loop polls settings at 1 Hz.
