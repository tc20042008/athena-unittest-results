import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle

def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']


class TestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y



class PrimitiveOp0(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp0(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([15, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([15, 25], dtype='int32'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp0()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp1(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([15, 25], dtype='int32'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp1()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp2(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, -1, -1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp2()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp3(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 15, 25], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, -1, -1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp3()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp4(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp4(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([6, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp4()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp5(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp5(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([6, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp5()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp6(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp6(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([100, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 152], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[100, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp6()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp7(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp7(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 152], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp7()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp8(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp8(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([50, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([50, 76], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[50, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp8()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp9(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp9(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([50, 76], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp9()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp10(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp10(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([25, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([25, 38], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[25, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp10()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp11(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp11(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([25, 38], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp11()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp12(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp12(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([13, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 19], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[13, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp12()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp13(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp13(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([13, 19], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp13()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp14(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp14(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([7, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 10], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[7, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp14()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp15(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp15(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([7, 10], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp15()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp16(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp16(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([6, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp16()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp17(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.expand(input_0, input_1)

class TestPrimitiveOp17(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([6, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp17()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()