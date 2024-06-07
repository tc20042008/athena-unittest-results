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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp0(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[1, 11109], dtype='int32'),
            paddle.to_tensor([1, 11109, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 11109], dtype='int32'),
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
        net = PrimitiveOp0()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp1(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([0], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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
        net = PrimitiveOp1()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp2(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([0], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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
        net = PrimitiveOp2()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp3(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.to_tensor([0], dtype='int32'),
            paddle.to_tensor([], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp4(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 80], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 100, 80], dtype='float32'),
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
        net = PrimitiveOp4()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp5(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp5(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 100, 4], dtype='float32'),
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
        net = PrimitiveOp5()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp6(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp6(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp7(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 68], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp8(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
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
        net = PrimitiveOp8()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp9(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp9(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
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
        net = PrimitiveOp9()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp10(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp10(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68, 68, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68, 68, 2], dtype='float32'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp11(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp12(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 34], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp13(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
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
        net = PrimitiveOp13()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp14(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp14(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
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
        net = PrimitiveOp14()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp15(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp15(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34, 34, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34, 34, 2], dtype='float32'),
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
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp16(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([17, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
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
        net = PrimitiveOp16()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp17(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp17(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 17], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
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
        net = PrimitiveOp17()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp18(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp18(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([17, 17, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
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
        net = PrimitiveOp18()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp19(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp19(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([17, 17, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
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
        net = PrimitiveOp19()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp20(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp20(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17, 17, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17, 17, 2], dtype='float32'),
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
        net = PrimitiveOp20()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp21(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp21(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp21()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp22(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp22(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 68], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp22()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp23(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp23(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
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
        net = PrimitiveOp23()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp24(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp24(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([68, 68, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
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
        net = PrimitiveOp24()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp25(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp25(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68, 68, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68, 68, 2], dtype='float32'),
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
        net = PrimitiveOp25()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp26(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp26(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
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
        net = PrimitiveOp26()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp27(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp27(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 34], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
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
        net = PrimitiveOp27()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp28(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp28(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
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
        net = PrimitiveOp28()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp29(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp29(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([34, 34, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
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
        net = PrimitiveOp29()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp30(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp30(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([34, 34, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[34, 34, 2], dtype='float32'),
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
        net = PrimitiveOp30()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp31(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp31(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([17, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
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
        net = PrimitiveOp31()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp32(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp32(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 17], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
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
        net = PrimitiveOp32()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp33(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp33(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([17, 17, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
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
        net = PrimitiveOp33()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp34(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp34(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([17, 17, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
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
        net = PrimitiveOp34()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp35(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp35(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([17, 17, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[17, 17, 2], dtype='float32'),
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
        net = PrimitiveOp35()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp36(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp36(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[240], dtype='float32'),
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
        net = PrimitiveOp36()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp37(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp37(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
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
        net = PrimitiveOp37()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp38(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp38(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 80, 169], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 13, 13], dtype='float32'),
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
        net = PrimitiveOp38()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp39(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp39(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 76, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 76, 169], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 76, 13, 13], dtype='float32'),
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
        net = PrimitiveOp39()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp40(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp40(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
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
        net = PrimitiveOp40()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp41(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp41(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 3549, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 3549, 4, 19], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 3549, 76], dtype='float32'),
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
        net = PrimitiveOp41()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp42(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp42(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp42()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp43(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp43(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp43()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp44(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp44(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 256, -1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
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
        net = PrimitiveOp44()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp45(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp45(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 100, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 100, 256], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 100, 256], dtype='float32'),
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
        net = PrimitiveOp45()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp46(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp46(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([960], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[960], dtype='float32'),
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
        net = PrimitiveOp46()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp47(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp47(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
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
        net = PrimitiveOp47()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp48(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp48(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 80, 2704], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
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
        net = PrimitiveOp48()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp49(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp49(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 76, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 76, 2704], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 76, 52, 52], dtype='float32'),
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
        net = PrimitiveOp49()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp50(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp50(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[192], dtype='float32'),
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
        net = PrimitiveOp50()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp51(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp51(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
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
        net = PrimitiveOp51()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp52(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp52(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
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
        net = PrimitiveOp52()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp53(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp53(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
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
        net = PrimitiveOp53()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp54(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp54(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.to_tensor([0, 0, 0], dtype='int32'),
            paddle.to_tensor([3, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
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
        net = PrimitiveOp54()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp55(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp55(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.to_tensor([0, 1, 2], dtype='int32'),
            paddle.to_tensor([3, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3], dtype='int32'),
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
        net = PrimitiveOp55()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp56(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp56(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 80], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 80], dtype='float32'),
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
        net = PrimitiveOp56()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp57(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp57(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 80, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3, 80], dtype='float32'),
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
        net = PrimitiveOp57()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp58(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp58(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2352], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 28, 28], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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
        net = PrimitiveOp58()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp59(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp59(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([3, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[3], dtype='float32'),
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
        net = PrimitiveOp59()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp60(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp60(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([91], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[91], dtype='float32'),
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
        net = PrimitiveOp60()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp61(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp61(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 15, 1024], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 15, 32, 32], dtype='float32'),
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
        net = PrimitiveOp61()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp62(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp62(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4, 1024], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4, 32, 32], dtype='float32'),
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
        net = PrimitiveOp62()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp63(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp63(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 91, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 91, 1024], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 91, 32, 32], dtype='float32'),
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
        net = PrimitiveOp63()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp64(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp64(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 11109, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 11109, 4, 17], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 11109, 68], dtype='float32'),
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
        net = PrimitiveOp64()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp65(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp65(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
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
        net = PrimitiveOp65()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp66(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp66(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
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
        net = PrimitiveOp66()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp67(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp67(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([24], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
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
        net = PrimitiveOp67()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp68(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp68(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[192], dtype='float32'),
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
        net = PrimitiveOp68()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp69(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp69(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp69()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp70(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp70(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp70()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp71(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp71(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp71()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp72(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp72(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp72()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp73(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp73(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp73()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp74(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp74(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp74()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp75(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp75(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp75()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp76(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp76(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp76()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp77(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp77(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([120], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[120], dtype='float32'),
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
        net = PrimitiveOp77()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp78(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp78(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 153450, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 153450, 1], dtype='float32'),
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
        net = PrimitiveOp78()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp79(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp79(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 153450, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 153450, 4], dtype='float32'),
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
        net = PrimitiveOp79()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp80(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp80(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([960], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[960], dtype='float32'),
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
        net = PrimitiveOp80()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp81(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp81(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
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
        net = PrimitiveOp81()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp82(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp82(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp82()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp83(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp83(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
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
        net = PrimitiveOp83()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp84(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp84(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp84()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp85(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp85(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 205923, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 205923, 1], dtype='float32'),
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
        net = PrimitiveOp85()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp86(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp86(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 205923, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 205923, 4], dtype='float32'),
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
        net = PrimitiveOp86()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp87(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp87(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp87()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp88(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp88(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 529], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, 23, 23], dtype='float32'),
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
        net = PrimitiveOp88()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp89(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp89(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 68, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 68, 529], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 68, 23, 23], dtype='float32'),
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
        net = PrimitiveOp89()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp90(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp90(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp90()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp91(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp91(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp91()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp92(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp92(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
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
        net = PrimitiveOp92()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp93(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp93(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([480], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[480], dtype='float32'),
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
        net = PrimitiveOp93()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp94(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp94(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 152, 272, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 128], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 152, 272, 128], dtype='float32'),
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
        net = PrimitiveOp94()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp95(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp95(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[1, 500], dtype='int64'),
            paddle.to_tensor([1, 500, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 500], dtype='int64'),
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
        net = PrimitiveOp95()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp96(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp96(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[1, 500], dtype='int32'),
            paddle.to_tensor([1, 500, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 500], dtype='int32'),
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
        net = PrimitiveOp96()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp97(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp97(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([0], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 128], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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
        net = PrimitiveOp97()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp98(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp98(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([115200, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[115200, 4], dtype='float32'),
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
        net = PrimitiveOp98()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp99(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp99(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([28800, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[28800, 4], dtype='float32'),
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
        net = PrimitiveOp99()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp100(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp100(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([7200, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[7200, 4], dtype='float32'),
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
        net = PrimitiveOp100()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp101(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp101(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1800, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1800, 4], dtype='float32'),
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
        net = PrimitiveOp101()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp102(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp102(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([450, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[450, 4], dtype='float32'),
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
        net = PrimitiveOp102()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp103(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp103(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 160, 240, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 160, 240, 3], dtype='float32'),
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
        net = PrimitiveOp103()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp104(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp104(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 120, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 120, 3], dtype='float32'),
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
        net = PrimitiveOp104()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp105(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp105(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 40, 60, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 40, 60, 3], dtype='float32'),
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
        net = PrimitiveOp105()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp106(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp106(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 20, 30, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 20, 30, 3], dtype='float32'),
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
        net = PrimitiveOp106()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp107(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp107(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 10, 15, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 10, 15, 3], dtype='float32'),
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
        net = PrimitiveOp107()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp108(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp108(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 160, 240, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 160, 240, 12], dtype='float32'),
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
        net = PrimitiveOp108()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp109(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp109(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 120, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 120, 12], dtype='float32'),
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
        net = PrimitiveOp109()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp110(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp110(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 40, 60, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 40, 60, 12], dtype='float32'),
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
        net = PrimitiveOp110()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp111(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp111(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 20, 30, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 20, 30, 12], dtype='float32'),
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
        net = PrimitiveOp111()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp112(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp112(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 10, 15, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 10, 15, 12], dtype='float32'),
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
        net = PrimitiveOp112()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp113(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp113(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([48], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
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
        net = PrimitiveOp113()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp114(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp114(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp114()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp115(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp115(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp115()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp116(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp116(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp116()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp117(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp117(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp117()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp118(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp118(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp118()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp119(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp119(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp119()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp120(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp120(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp120()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp121(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp121(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp121()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp122(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp122(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp122()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp123(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp123(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp123()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp124(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp124(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
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
        net = PrimitiveOp124()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp125(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp125(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([258], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[258], dtype='float32'),
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
        net = PrimitiveOp125()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp126(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp126(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([258], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[258], dtype='float32'),
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
        net = PrimitiveOp126()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp127(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp127(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([258], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[258], dtype='float32'),
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
        net = PrimitiveOp127()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp128(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp128(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
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
        net = PrimitiveOp128()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp129(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp129(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp129()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp130(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp130(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([91], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[91], dtype='float32'),
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
        net = PrimitiveOp130()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp131(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp131(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 15, 16384], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 15, 128, 128], dtype='float32'),
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
        net = PrimitiveOp131()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp132(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp132(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4, 16384], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4, 128, 128], dtype='float32'),
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
        net = PrimitiveOp132()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp133(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp133(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 91, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 91, 16384], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 91, 128, 128], dtype='float32'),
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
        net = PrimitiveOp133()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp134(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp134(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 80, 289], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 17, 17], dtype='float32'),
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
        net = PrimitiveOp134()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp135(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp135(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4, 289], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4, 17, 17], dtype='float32'),
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
        net = PrimitiveOp135()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp136(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp136(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 289], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, 17, 17], dtype='float32'),
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
        net = PrimitiveOp136()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp137(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp137(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp137()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp138(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp138(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp138()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp139(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp139(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 80, 4624], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 68, 68], dtype='float32'),
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
        net = PrimitiveOp139()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp140(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp140(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4, 4624], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4, 68, 68], dtype='float32'),
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
        net = PrimitiveOp140()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp141(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp141(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 4624], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, 68, 68], dtype='float32'),
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
        net = PrimitiveOp141()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp142(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp142(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([258], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[258], dtype='float32'),
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
        net = PrimitiveOp142()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp143(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp143(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([258], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[258], dtype='float32'),
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
        net = PrimitiveOp143()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp144(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp144(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([258], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[258], dtype='float32'),
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
        net = PrimitiveOp144()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp145(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp145(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([154560, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[154560, 4], dtype='float32'),
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
        net = PrimitiveOp145()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp146(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp146(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([38640, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[38640, 4], dtype='float32'),
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
        net = PrimitiveOp146()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp147(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp147(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([9660, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[9660, 4], dtype='float32'),
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
        net = PrimitiveOp147()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp148(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp148(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([2415, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[2415, 4], dtype='float32'),
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
        net = PrimitiveOp148()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp149(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp149(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([648, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[648, 4], dtype='float32'),
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
        net = PrimitiveOp149()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp150(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp150(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 184, 280, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 184, 280, 3], dtype='float32'),
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
        net = PrimitiveOp150()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp151(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp151(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 92, 140, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 92, 140, 3], dtype='float32'),
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
        net = PrimitiveOp151()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp152(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp152(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 46, 70, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 46, 70, 3], dtype='float32'),
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
        net = PrimitiveOp152()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp153(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp153(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 23, 35, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 23, 35, 3], dtype='float32'),
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
        net = PrimitiveOp153()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp154(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp154(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 12, 18, 3], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 12, 18, 3], dtype='float32'),
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
        net = PrimitiveOp154()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp155(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp155(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 184, 280, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 184, 280, 12], dtype='float32'),
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
        net = PrimitiveOp155()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp156(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp156(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 92, 140, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 92, 140, 12], dtype='float32'),
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
        net = PrimitiveOp156()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp157(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp157(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 46, 70, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 46, 70, 12], dtype='float32'),
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
        net = PrimitiveOp157()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp158(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp158(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 23, 35, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 23, 35, 12], dtype='float32'),
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
        net = PrimitiveOp158()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp159(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp159(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 12, 18, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 12, 18, 12], dtype='float32'),
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
        net = PrimitiveOp159()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp160(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp160(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([91], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[91], dtype='float32'),
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
        net = PrimitiveOp160()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp161(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp161(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 15, 4096], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 15, 64, 64], dtype='float32'),
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
        net = PrimitiveOp161()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp162(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp162(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4, 4096], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4, 64, 64], dtype='float32'),
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
        net = PrimitiveOp162()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp163(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp163(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 91, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 91, 4096], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 91, 64, 64], dtype='float32'),
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
        net = PrimitiveOp163()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp164(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp164(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
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
        net = PrimitiveOp164()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp165(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp165(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 80, 676], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 26, 26], dtype='float32'),
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
        net = PrimitiveOp165()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp166(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp166(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 76, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 76, 676], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 76, 26, 26], dtype='float32'),
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
        net = PrimitiveOp166()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp167(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp167(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([480], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[480], dtype='float32'),
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
        net = PrimitiveOp167()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp168(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp168(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.randint(low=0, high=1, shape=[1, 3549], dtype='int32'),
            paddle.to_tensor([1, 3549, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
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
        net = PrimitiveOp168()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp169(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp169(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([0], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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
        net = PrimitiveOp169()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp170(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp170(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([0], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
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
        net = PrimitiveOp170()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp171(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp171(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.to_tensor([0], dtype='int32'),
            paddle.to_tensor([], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = PrimitiveOp171()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp172(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp172(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[384], dtype='float32'),
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
        net = PrimitiveOp172()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp173(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp173(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp173()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp174(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp174(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
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
        net = PrimitiveOp174()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp175(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp175(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
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
        net = PrimitiveOp175()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp176(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp176(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 32], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
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
        net = PrimitiveOp176()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp177(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp177(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
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
        net = PrimitiveOp177()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp178(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp178(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([32, 32, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
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
        net = PrimitiveOp178()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp179(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp179(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[32, 32, 2], dtype='float32'),
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
        net = PrimitiveOp179()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp180(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp180(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
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
        net = PrimitiveOp180()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp181(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp181(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 64], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
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
        net = PrimitiveOp181()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp182(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp182(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
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
        net = PrimitiveOp182()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp183(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp183(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([64, 64, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
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
        net = PrimitiveOp183()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp184(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp184(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[64, 64, 2], dtype='float32'),
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
        net = PrimitiveOp184()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp185(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp185(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
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
        net = PrimitiveOp185()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp186(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp186(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 128], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
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
        net = PrimitiveOp186()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp187(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp187(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
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
        net = PrimitiveOp187()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp188(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp188(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([128, 128, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
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
        net = PrimitiveOp188()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp189(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp189(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 2], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128, 128, 2], dtype='float32'),
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
        net = PrimitiveOp189()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp190(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp190(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp190()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp191(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp191(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp191()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp192(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp192(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp192()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp193(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp193(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp193()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp194(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp194(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp194()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp195(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp195(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp195()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp196(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp196(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp196()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp197(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp197(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp197()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp198(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp198(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp198()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp199(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp199(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 512, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
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
        net = PrimitiveOp199()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp200(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp200(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
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
        net = PrimitiveOp200()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp201(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp201(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp201()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp202(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp202(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([240], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[240], dtype='float32'),
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
        net = PrimitiveOp202()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp203(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp203(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
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
        net = PrimitiveOp203()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp204(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp204(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
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
        net = PrimitiveOp204()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp205(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp205(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 100, 4], dtype='float32'),
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
        net = PrimitiveOp205()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp206(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp206(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 1, 4], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
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
        net = PrimitiveOp206()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp207(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp207(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 80, 1156], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 80, 34, 34], dtype='float32'),
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
        net = PrimitiveOp207()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp208(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp208(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 4, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 4, 1156], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 4, 34, 34], dtype='float32'),
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
        net = PrimitiveOp208()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp209(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp209(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 1156], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, 34, 34], dtype='float32'),
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
        net = PrimitiveOp209()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp210(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp210(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp210()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp211(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp211(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 8464], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, 92, 92], dtype='float32'),
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
        net = PrimitiveOp211()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp212(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp212(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 68, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 68, 8464], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 68, 92, 92], dtype='float32'),
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
        net = PrimitiveOp212()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp213(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp213(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([100, 256, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([100, 256, -1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
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
        net = PrimitiveOp213()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp214(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp214(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 100, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 100, 256], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 100, 256], dtype='float32'),
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
        net = PrimitiveOp214()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp215(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp215(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp215()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp216(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp216(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 128, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[128], dtype='float32'),
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
        net = PrimitiveOp216()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp217(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp217(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
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
        net = PrimitiveOp217()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp218(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp218(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
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
        net = PrimitiveOp218()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp219(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp219(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
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
        net = PrimitiveOp219()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp220(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp220(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[384], dtype='float32'),
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
        net = PrimitiveOp220()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp221(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp221(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp221()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp222(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp222(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 1, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 1, 2116], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 1, 46, 46], dtype='float32'),
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
        net = PrimitiveOp222()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp223(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp223(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([1, 68, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 68, 2116], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[1, 68, 46, 46], dtype='float32'),
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
        net = PrimitiveOp223()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp224(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp224(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[192], dtype='float32'),
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
        net = PrimitiveOp224()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp225(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp225(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
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
        net = PrimitiveOp225()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp226(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp226(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([68], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
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
        net = PrimitiveOp226()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp227(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp227(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([27], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[27], dtype='float32'),
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
        net = PrimitiveOp227()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp228(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp228(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, 256, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
        net = PrimitiveOp228()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp229(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp229(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[384], dtype='float32'),
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
        net = PrimitiveOp229()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp230(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp230(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
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
        net = PrimitiveOp230()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

class PrimitiveOp231(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.reshape(input_0, input_1), None

class TestPrimitiveOp231(TestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([1, -1, 1, 1], dtype='int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            paddle.static.InputSpec(shape=[192], dtype='float32'),
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
        net = PrimitiveOp231()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()