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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp0(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp1(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp2(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp3(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp4(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp5(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp6(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 43, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp7(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 43, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp8(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp9(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp10(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp11(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp12(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp13(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 4, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp14(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp15(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp16(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp17(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 11, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
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

class PrimitiveOp18(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp18(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 11, 12, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp18()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp19(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp19(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[49, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp19()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp20(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp20(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 11, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp20()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp21(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp21(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 11, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp21()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp22(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp22(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp22()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp23(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp23(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp23()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp24(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp24(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 16, 6, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp24()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp25(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp25(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[16, 16], dtype='int64'),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp25()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp26(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp26(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[49, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp26()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp27(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp27(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp27()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp28(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp28(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp28()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp29(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp29(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 54, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp29()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp30(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp30(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp30()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp31(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp31(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp31()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp32(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp32(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 11, 1, 24, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp32()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp33(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp33(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[16, 49], dtype='int64'),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[16, 49], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp33()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp34(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp34(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp34()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp35(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp35(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp35()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp36(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp36(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 86, 3, 197, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp36()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp37(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp37(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp37()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp38(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp38(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp38()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp39(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp39(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp39()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp40(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp40(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2, 43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2, 43, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp40()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp41(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp41(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp41()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp42(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp42(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1], dtype='int64'),
            paddle.to_tensor([2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp42()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp43(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp43(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([2], dtype='int64'),
            paddle.to_tensor([3], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 43, 64, 3, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp43()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp44(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle.slice(input_0, axes=[0], starts=input_1, ends=input_2)

class TestPrimitiveOp44(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[196, 196], dtype='int64'),
            paddle.to_tensor([0], dtype='int64'),
            paddle.to_tensor([1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp44()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()