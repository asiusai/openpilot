## ledd

LEDs are the primary way of communicating the device state to the user. LEDs are on the back side of the 2 road facing cameras, each has 3 individual LEDs. They are used so that the LEDs on the sides (so LEDs 1, 3, 4 and 6) brightness is derived from how bright the outside is and the LEDs on the middle (so 2 and 5) are 10% of that, this is just so it looks nicer. Max brightness is 50% and min is 10%. The current brightness should be changed based on the exposure of the wide road camera. During startup or when we dont know the exposure, it should be set to 10%.

| color               | meaning                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| WHITE smooth fading | starting up                                                                 |
| WHITE               | running, normal state                                                       |
| GREEN               | engaged                                                                     |
| BLUE blinking       | bluetooth pairing mode                                                      |
| ORANGE progress     | calibration, LED 1 always on, others advance with calibration progress      |
| RED                 | device/system fault or error                                                |
| RED blinking        | error while engaged, for example soon will disengage, be ready to take over |
| MAGNETA blinking    | driver monitoring error, pay attention or driver not detected               |
