# Asius v0 LEDs

| Appearance | Meaning |
| --- | --- |
| Fading white | Starting up |
| White | Running normally, disengaged |
| Green | Engaged |
| Blinking blue | Bluetooth pairing |
| Orange progress | Calibrating, with LED 1 on from 0% |
| Red | Device or system fault |
| Blinking red | Fault or driving warning while engaged |
| Blinking magenta | Driver-monitoring alert or driver not detected while onroad |

LEDs are numbered 1 through 6, left to right when facing the light windows.
LEDs 1-3 are wide-camera packages 2, 1, 0; LEDs 4-6 are road-camera packages
2, 1, 0. The gap from 3 to 4 is 30 mm; the gaps from 1 to 3 and 4 to 6 are
15 mm. `led_patterns.py` owns the logical-to-hardware mapping.

The OS smoothly fades all six warm-white LEDs in and out together during boot.
`ledd` continues the same three-second curve during startup, then takes over with
steady warm white. Kernel startup peaks below 10% before camera exposure is
available. Automatic lighting uses the wide road camera's exposure to choose
an outer-LED peak from 5% in the dark to 49% in bright conditions. Missing,
invalid or stale exposure readings use the same 10% fallback as boot.
The fade, blink and calibration progress phases can go below that peak or off.
LEDs 2 and 5 sit midway
across the 15 mm gaps and always run at 10% output, including startup,
calibration, warning indications and manual control.

Calibration fills LEDs 1 through 6 in deep orange using fresh
`extrinsicsCalibration.calPerc`: the four outer LEDs mark 25% intervals, with
the two center LEDs filling halfway between their neighbors. LED 1 stays orange
from 0% so calibration is visible before progress begins. Completion returns
directly to normal status. There is no success flash. Faults and engaged warnings
take priority over calibration.

Engagement blockers, including a held brake or pre-enabled state, leave the LEDs
white. Faults are also shown while parked; processes intentionally stopped while
parked are not faults. A fault blinks red when engaged. Driver monitoring uses
fresh valid data; missing drivers also blink magenta before engagement, but
an intentionally disabled driver camera does not raise that warning.

Manual control remains parked-only, retains its full 0%-100% range, and addresses all six LEDs in the
same logical order as the app. Bluetooth pairing takes priority over manual
control; manual overrides clear when driving begins.
The app reports the daemon's current automatic brightness instead of a fixed
10% default. This is the selected peak brightness, not an animation's instant value.

The VamOS kernel table and this startup curve should stay in sync. Run the LED
tests with `python -m pytest openpilot/selfdrive/v0/tests/test_led*.py`.
