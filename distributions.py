import mxnet as mx
from mxnet import np, npx
from mxnet import gluon
npx.set_np()


def getF(var):
    if isinstance(var, np.ndarray):
        return np.ndarray
    elif isinstance(var, mx.symbol.numpy._Symbol):
        return mx.symbol.numpy._Symbol
    else:
        raise RuntimeError("var must be instance of NDArray or Symbol in getF")


class Normal():
  def __init__(self, loc, scale, F=None):
    self._loc = loc
    self._scale = scale
    self.F = F

  def sample(self, size=None):
    return self.F.np.random.normal(self._loc,
                                   self._scale,
                                   size)

  def sample_n(self, batch_size=None):
    return self.F.npx.random.normal_n(self._loc,
                                      self._scale,
                                      batch_size)

  def log_prob(self, value):
    F = self.F
    var = (self._scale ** 2)
    log_scale = F.np.log(self._scale)
    return -((value - self._loc) ** 2) / (2 * var) - log_scale - F.np.log(F.np.sqrt(2 * F.np.pi))


# This works, yeah!
class NormalTest(gluon.HybridBlock):
    def __init__(self):
      super(NormalTest, self).__init__()

    def hybrid_forward(self, F, loc, scale):
      mvn = Normal(loc, scale, F)
      return mvn.sample()


# normal_test = NormalTest()
# normal_test.hybridize()
# print(normal_test(np.ones((2,2)), np.ones((2,2))))
