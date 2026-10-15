# Bundled speech

`soundd` uses this directory for all spoken clips. Playback works without internet.
`messages.json` contains every phrase and digit. `generated.json` contains their generation settings and hashes.
Keep each WAV file beside these files. Use the catalog key as its filename.

The catalog contains 13 alert phrases and 17 pairing clips.
Pairing keys use the `pairing_` prefix. They include ten digits, instructions and results.
Pairing speech works when `VoiceAlertsEnabled` is off.
Digit clips use 1.25x playback tempo with unchanged pitch. Instructions retain their normal tempo.

Enable optional screen alerts and offroad fault speech through the Asius app's Controls page.
The persistent `VoiceAlertsEnabled` toggle defaults to off.
Alert entries contain their event and offroad-parameter bindings.
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

Generate speech only during development. Device builds validate the committed files without TTS dependencies or network access.
`scons speech` checks every clip in the shared catalog. The normal build runs the same check.
Tests reject missing clips, extra clips, damaged audio and text changes without regenerated files.

Processing uses the louder asius-1 speech test settings: loudness normalization,
speech-band filtering, compression and a peak limiter. Keep the Dragon mixer
at 8192, its normal unity gain. The maximum mixer value adds almost 8x gain and
clips these files. This is a tested playback setting, not a calibrated SPL rating.

Speech reserves 15% of full scale for existing noncritical chimes. Critical
sirens interrupt speech and retain their original samples, ramp and escalation.
The callback only reads preloaded samples. It never synthesizes, decodes files,
reads parameters or starts a subprocess. The control loop polls settings at 1 Hz.
