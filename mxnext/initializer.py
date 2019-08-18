# encoding: utf-8
"""
MXNeXt is a wrapper around the original MXNet Symbol API
@version: 0.1
@author:  Yuntao Chen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import mxnet as mx
import numpy as np
from mxnet.initializer import Initializer, register


def gauss(std):
    return mx.init.Normal(sigma=std)


def one_init():
    return mx.init.One()


def zero_init():
    return mx.init.Zero()


def constant(val):
    return mx.init.Constant(val)


@register
class Custom(Initializer):
    def __init__(self, arr):
        super(Custom, self).__init__(arr=arr)
        self.arr = arr

    def _init_weight(self, _, arr):
        val = mx.nd.array(self.arr)
        assert val.size == arr.size, "init shape {} is not compatible with weight shape {}".format(val.shape, arr.shape)
        arr[:] = val.reshape_like(arr)


def custom(arr):
    return Custom(arr)
