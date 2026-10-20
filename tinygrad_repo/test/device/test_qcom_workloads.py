import math, unittest

from tinygrad import Context, Device, Tensor, nn


@unittest.skipUnless(Device.DEFAULT.split(":")[0] == "QCOM", "requires a QCOM device")
class TestQCOMWorkloads(unittest.TestCase):
  def test_mnist_batch32_adam_step(self):
    # Covers convolution forward/backward plus optimizer updates. This shape previously generated
    # IR3 kernels with masked-axis and full-reduction unrolls that watchdog Adreno 6xx.
    from examples.beautiful_mnist import Model

    Tensor.manual_seed(0)
    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    opt.zero_grad()
    with Context(TRAINING=1):
      loss = model(Tensor.rand(32, 1, 28, 28)).sparse_categorical_crossentropy(Tensor.randint(32, high=10)).backward()
      loss.realize(*opt.schedule_step())

    self.assertTrue(math.isfinite(loss.item()))


if __name__ == "__main__": unittest.main()
