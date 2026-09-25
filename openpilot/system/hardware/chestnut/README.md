# Chestnut firmware

The bundled image is `custom ed4e39b7-quiet1`, built from
[tinygrad/asm2464pd-firmware](https://github.com/tinygrad/asm2464pd-firmware)
commit `ed4e39b7e0794e19ba193477067c48757a5cf9ef` plus [firmware.patch](firmware.patch).
Its SHA256 is
`e9523631afd644104eb48b2ad3590c83dd1e13fceea2f529a0d0aa0e1ea87b14`.

Apply the patch to that revision, then build with SDCC and binutils:

```sh
git apply /path/to/openpilot/system/hardware/chestnut/firmware.patch
make -C handmade wrapped
sha256sum handmade/build/firmware_wrapped.bin
```

The firmware parks downstream PCIe on the observed SuperSpeed host-loss event
and leaves it off until tinygrad's F3 request. Idle USB2 fallback triggers a
20-second retry, limited to two attempts across CPU resets. GPU/flash requests
cancel recovery. F4 (wValue=1, wIndex=0, wLength=0) atomically selects USB3 mode
and restarts the CPU. Loading newly flashed code from SPI still requires a
physical reset or power cycle; a CPU restart retains the running RAM image.

The flasher and device selection check the complete build identifier. When GPU
models exist, startup waits up to 45 seconds for a detected Chestnut to reach
SuperSpeed before GPU initialization can cancel its retry. Missing devices
return immediately.

With the VamOS USB recovery kernel, bench tests passed 55/55 host reboots,
10/10 SuperSpeed port reconnects, and 8 GiB of verified GPU transfers each way.
USB2 retries stopped after two attempts and active USB2 work cancelled them.
The debug cable remained attached; removal, both Type-C orientations, other
GPUs and a full Openpilot run remain unverified for this build.
