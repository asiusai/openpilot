## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

Asius builds `big_driving_asius_tinygrad.pkl` on the attached Chestnut GPU.
The upstream Cinque v3 pickle contains gfx12 machine code and cannot execute
on the bench TinyChestnut's gfx1100 GPU. Until a compatible v3 source model
is available, Asius retains the last public Cinque v2 ONNX from upstream
`81ae1a2e2dbd7147544732acca32c4db9cf10b0f`, with SHA256
`6fee5937923c74848df4a63f6239eb6331c6274dd4bdb7a5d6ec0388a8b543d5`.
It includes the history queues required by the current model runner.
The native build runs the compiler's output and pickle reload checks.
