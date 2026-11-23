## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

The big-model artifacts use tinygrad `7dc5a00f4c15f7e66eb69decfa44faae14a65ad1` serialization.
They retain the GPU kernels and weights from openpilot `af90646a4`.
The export updates operation identifiers, native-function nodes, external runtime buffers, call metadata, and captured-JIT arguments.
CPU helpers remain portable and compile when loaded.

The unchanged big-model weight payload has SHA-256 `76a5c0712e4a9fd2c8bb95d1a562661f0b668c309cb7d4ac03ac7a6f79fc8c94`.
