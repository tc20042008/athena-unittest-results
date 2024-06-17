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

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

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

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

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



class PrimitiveOp_151b2d2c06500a70afd58350de35f3be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.roll(input_0, input_1, [1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c674c6486e6d4fd063ae20c6446c09fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([43, 56, 56, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0406b97489ac3fc2df12eb4ec44f6db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_08496eb2516973a9ed5ceec2f0cc693d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_89df091b00fd5504fbb433ed4195c129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3134d59298e765e91c1c2f122032e227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([43, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_04bc8e0dd5846fd33ceaf524daf86929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_45ba74358f39a5edf855c557e44e577b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_f7d51451bc5934c3a9996ca68c9c0def(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 28, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0406b97489ac3fc2df12eb4ec44f6db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_89df091b00fd5504fbb433ed4195c129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_151b2d2c06500a70afd58350de35f3be
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-3, -3], dtype='int64').reshape([2]),
        ]




if __name__ == '__main__':
    unittest.main()