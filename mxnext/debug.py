"""
Debug Op
author: Yuntao Chen
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import marshal
import types
import uuid

import mxnet as mx
import numpy as np


__all__ = ["forward_debug", "backward_debug"]


np.set_printoptions(threshold=np.inf, precision=4, linewidth=120)


class DebugOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(DebugOperator, self).__init__()
        self.pos = kwargs.get("pos", None)
        self.num_args = int(kwargs.get("num_args", 1))
        self.do_forward = bool(kwargs.get("do_forward", False))
        self.do_backward = bool(kwargs.get("do_backward", False))
        if "callback" in kwargs:
            callback_func_code = marshal.loads(json.loads(kwargs["callback"]).encode("latin"))
            self.callback = types.FunctionType(callback_func_code, globals())

    def forward(self, is_train, req, in_data, out_data, aux):
        if self.do_forward:
            aux[0] += 1
            in_data_cpu = [aux[0].context.device_id] + [aux[0].asscalar()] + [_.asnumpy() for _ in in_data]
            self.callback(*in_data_cpu)

        for o, r, i in zip(out_data, req, in_data):
            self.assign(o, r, i)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.do_backward:
            aux[0] += 1
            out_grad_cpu = [aux[0].context.device_id] + [aux[0].asscalar()] + [_.asnumpy() for _ in out_grad]
            self.callback(*out_grad_cpu)

        for i, r, o in zip(in_grad, req, out_grad):
            self.assign(i, r, o)


@mx.operator.register("Debug")
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(DebugProp, self).__init__(need_top_grad=True)
        self._kwargs = kwargs

    def list_arguments(self):
        inputs = ['data']
        num_args = int(self._kwargs.get("num_args", 1))
        for i in range(1, num_args):
            inputs.append("data%d" % i)
        return inputs

    def list_auxiliary_states(self):
        return ["num_iter"]

    def list_outputs(self):
        outputs = ['output']
        num_args = int(self._kwargs.get("num_args", 1))
        for i in range(1, num_args):
            outputs.append("output%d" % i)
        return outputs

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return out_grad

    def infer_shape(self, in_shape):
        return in_shape, in_shape, [(1, )]

    def create_operator(self, ctx, shapes, dtypes):
        return DebugOperator(**self._kwargs)


def Debug(data, type="nchw", num_args=1, **kwargs):
    kwargs.update({"pos": type, "data": data, "num_args": num_args, "do_forward": True})
    return mx.sym.Custom(op_type="Debug", **kwargs)


def default_print_function(*inputs):
    for i, input in enumerate(inputs[1:]):
        print("input{}: {}".format(i, input))


def forward_debug(*data, **kwargs):
    kwargs.update({"data": data[0], "num_args": len(data), "do_forward": True})
    # if len(input_symbols) > 1, give them names
    for i, v in enumerate(data[1:], start=1):
        kwargs.update({"data%d" % i: v})
    callback = kwargs.get("callback", default_print_function)
    callback_code = marshal.dumps(callback.__code__)
    kwargs["callback"] = json.dumps(callback_code.decode("latin"))
    num_iter = mx.sym.var("num_iter_{}".format(uuid.uuid4()), init=mx.init.Constant(0))
    return mx.sym.Custom(op_type="Debug", num_iter=num_iter, **kwargs)


def backward_debug(*data, **kwargs):
    kwargs.update({"data": data[0], "num_args": len(data), "do_backward": True})
    # if len(input_symbols) > 1, give them names
    for i, v in enumerate(data[1:], start=1):
        kwargs.update({"data%d" % i: v})
    callback = kwargs.get("callback", default_print_function)
    callback_code = marshal.dumps(callback.__code__)
    kwargs["callback"] = json.dumps(callback_code.decode("latin"))
    num_iter = mx.sym.var("num_iter_{}".format(uuid.uuid4()), init=mx.init.Constant(0))
    return mx.sym.Custom(op_type="Debug", num_iter=num_iter, **kwargs)


if __name__ == "__main__":
    x = mx.sym.var("x")
    y = mx.sym.var("y")
    x = forward_debug(x)
    x = forward_debug(x, callback=lambda _, a: print("a forward = {}".format(a)))
    x = backward_debug(x, callback=lambda _, a: print("a backward = {}".format(a)))
    (x, y) = forward_debug(x, y, callback=lambda _, a, b: print("a + b = {}".format(a + b)))
    (x, y) = backward_debug(x, y, callback=lambda _, a, b: print("a + b backward = {}".format(a + b)))
    x = x * 2
    z = x + y  # type: mx.sym.Symbol

    print(z.get_internals())

    exe = z.simple_bind(mx.cpu(), x=(2, ), y=(2, ), grad_req="write")
    exe.forward(x=mx.nd.ones(2), y=mx.nd.full(2, val=2))
    exe.backward(mx.nd.ones(2))
    exe.forward(x=mx.nd.ones(2), y=mx.nd.full(2, val=2))
    exe.backward(mx.nd.ones(2))
    exe.forward(x=mx.nd.ones(2), y=mx.nd.full(2, val=2))
    exe.backward(mx.nd.ones(2))
    exe.outputs[0].wait_to_read()
    exe.grad_arrays[0].wait_to_read()
