# Chestnut firmware

The bundled image is `custom 8516b768-quiet3`, built from
[tinygrad/asm2464pd-firmware](https://github.com/tinygrad/asm2464pd-firmware)
commit `8516b7687318c087cb1c0493b2af4923691a4511` plus [firmware.patch](firmware.patch).
Its SHA256 is
`186d391a9d2a480ca49add4480a1f526036095fa59cece766751dcebfc40fb94`.

This revision includes the upstream [failed PCIe read fix](https://github.com/tinygrad/asm2464pd-firmware/pull/87).
Failed reads return a zero-length USB response, so the host rejects missing data.
The Asius USB recovery changes remain unchanged. Physical testing of `quiet3` is pending.
Two clean builds produced the same SHA256. The compiled firmware passed 80 emulator cases covering USB2, USB3 and failed PCIe reads.
These checks verify response length and transfer cancellation. They do not establish hardware recovery reliability.

Apply the patch to that revision, then build with SDCC and binutils:

```sh
git apply /path/to/openpilot/system/hardware/chestnut/firmware.patch
make -C handmade wrapped
sha256sum handmade/build/firmware_wrapped.bin
```

The firmware parks downstream PCIe on the observed SuperSpeed host-loss event
and leaves it off until tinygrad's F3 request. Idle USB2 fallback triggers a
20-second retry, starting at fallback even if USB2 never enumerates. A later
USB configuration restarts the timer. Recovery is limited to two attempts
across CPU resets. GPU/flash requests cancel recovery.
F4 (wValue=1, wIndex=0, wLength=0) atomically selects USB3 mode
and restarts the CPU. Loading newly flashed code from SPI still requires a
physical reset or power cycle; a CPU restart retains the running RAM image.

The flasher and device selection check the complete build identifier. When GPU
models exist, startup waits up to 45 seconds for a detected Chestnut to reach
SuperSpeed before GPU initialization can cancel its retry. Missing devices
return immediately.

With the VamOS USB recovery kernel on Dragon and a 7900 XTX, `quiet2`
recovered to SuperSpeed after an initial failure with USB2 deliberately
disabled. The previous `quiet1` build stayed absent in that fixture because
its timer required USB2 configuration. Three warm reboots and two Dragon
supply cycles also passed, with 2.625 GiB of verified GPU transfers each way
across the initial transfer test and restart tests. USB2-only retries stopped
after two attempts and the connection stayed usable for the rest of a
100-second observation. F3 GPU initialization canceled pending retries for
75 seconds, followed by 32 MiB of verified USB2 GPU transfers each way.

These tests used one Type-C orientation with the debug cable attached.
No debug reset or main-cable recovery command was used during the five
restart tests. Debug removal, other orientations, other GPUs and a full
Openpilot run remain unverified in these `quiet2` tests.
