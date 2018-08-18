import mxnet as mx
import mxnet.ndarray as nd

from mxnet.gluon import Parameter

from common import check_gradient


def test_check_gradient():
    # test check gradient on a simple function
    ctx = mx.cpu()
    w = Parameter(name='w', shape=(2, 3))
    w.initialize('zeros', ctx)
    w.set_data(nd.array([[1., 2., 3], [-1., -3., 1.5]]))

    def f():
        return nd.sum(nd.square(w.data()))

    assert check_gradient(f, [], w)
