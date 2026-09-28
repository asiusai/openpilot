# Asius v0 LEDs

LEDs are numbered 1 through 6, left to right when facing the light windows.
LEDs 1-3 are wide-camera packages 2, 1, 0; LEDs 4-6 are road-camera packages
2, 1, 0. The gap from 3 to 4 is 30 mm; the gaps from 1 to 3 and 4 to 6 are
15 mm. `led_patterns.py` owns the logical-to-hardware mapping.

The OS shows the warm-white Knight Rider sweep during boot. `ledd` continues
the same three-second curve during startup, then takes over with steady warm
white. Startup uses the approved full-brightness curve. Runtime status lighting
retains the camera-exposure brightness adjustment. LEDs 2 and 5 sit midway
across the 15 mm gaps and always run at 10% output, including startup,
calibration, warning indications and manual control.

Calibration fills LEDs 1 through 6 in deep orange using fresh
`extrinsicsCalibration.calPerc`: the four outer LEDs mark 25% intervals, with
the two center LEDs filling halfway between their neighbors. Completion returns
directly to normal status. There is no success flash. Faults and engaged warnings
take priority over calibration.

Engaged, unavailable and warning status colors otherwise retain their existing
meanings. Manual control remains parked-only and addresses all six LEDs in the
same logical order as the app. Bluetooth pairing takes priority over manual
control; manual overrides clear when driving begins.

The VamOS kernel table and this startup curve should stay in sync. Run the LED
tests with `python -m pytest openpilot/selfdrive/v0/tests/test_led*.py`.
