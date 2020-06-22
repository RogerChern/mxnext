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


from mxnext.initializer import one_init, zero_init


import mxnet as mx
import numpy as np


variable_registry = {}
def shared_var(name, **kwargs):
    if name not in variable_registry:
        var = mx.sym.var(name, **kwargs)
        variable_registry[name] = var

    return variable_registry[name]


def relu(data, name=None):
    """
    the linear rectifier activation function, y = max(x, 0)
    :param data: input symbol object
    :param name: output symbol name
    :return: output symbol object
    """
    if name is None:
        prev_name = data.name
        if prev_name.endswith("_bn") or prev_name.endswith("_gn"):
            name = prev_name[:-3] + "_relu"
        else:
            name = prev_name + "_relu"

    return mx.sym.Activation(data, name=name, act_type='relu')


def relu6(data, name=None):
    """
    the linear rectifier activation function, y = min(max(x, 0), 6)
    :param data: input symbol object
    :param name: output symbol name
    :return: output symbol object
    """
    if name is None:
        prev_name = data.name
        if prev_name.endswith("_bn") or prev_name.endswith("_gn"):
            name = prev_name[:-3] + "_relu"
        else:
            name = prev_name + "_relu"

    return mx.sym.clip(data, a_max=6, a_min=0, name=name)


def sigmoid(data, name=None):
    """
    the sigmoid activation function, y = 1 / (1 + exp(-x))
    :param data: input symbol object
    :param name: output symbol name
    :return: output symbol object
    """
    if name is None:
        prev_name = data.name
        if prev_name.endswith("_bn") or prev_name.endswith("_gn"):
            name = prev_name[:-3] + "_sigmoid"
        else:
            name = prev_name + "_sigmoid"

    return mx.sym.Activation(data, name=name, act_type='sigmoid')


def dropout(data, p_drop=0.5, name=None):
    """
    - During training, each element of the input is set to zero with probability p.
      The whole array is rescaled by 1/(1−p) to keep the expected sum of the input unchanged.
    - During testing, this operator does not change the input if mode is ‘training’.
      If mode is ‘always’, the same computaion as during training will be applied.
    :param data: input symbol object
    :param name: output symbol name
    :param p_drop: probability of each element set to zero
    :return: output symbol object
    """
    if name is None:
        prev_name = data.name
        name = prev_name + "_dropout"

    return mx.sym.Dropout(data, name=name, p=p_drop)


def roi_align(data, rois, name, out_size, stride):
    if isinstance(out_size, int):
        out_size = (out_size, out_size)

    return mx.sym.contrib.ROIAlign_v2(name=name,
                                      data=data,
                                      rois=rois,
                                      pooled_size=out_size,
                                      spatial_scale=1.0 / stride)


def concat(inputs, name, axis=1):
    assert isinstance(inputs, list), "Concat accepts a list of symbols"

    if len(inputs) == 1:
        return inputs[0]

    assert len(inputs) > 1, "Concat accepts > 1 symbols"

    return mx.sym.concat(*inputs, name=name, dim=axis)


def whiten(data, name="bn_data"):
    return mx.sym.BatchNorm(data=data,
                            name=name,
                            fix_gamma=True,
                            use_global_stats=True,
                            eps=1e-5+1e-10)


def conv(data, name, filter, kernel=1, stride=1, pad=None, dilate=1, num_group=1,
         no_bias=True, init=None, lr_mult=1.0, wd_mult=1.0, weight=None, bias=None):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilate, int):
        dilate = (dilate, dilate)
    if pad is None:
        assert kernel[0] % 2 == 1, "Specify pad for an even kernel size for {}".format(name)
        assert kernel[1] % 2 == 1, "Specify pad for an even kernel size for {}".format(name)
        pad = (((kernel[0] - 1) * dilate[0] + 1) // 2, ((kernel[1] - 1) * dilate[1] + 1) // 2)
    if isinstance(pad, int):
        pad = (pad, pad)

    # specific initialization method
    if not isinstance(weight, mx.sym.Symbol):
        if init is not None:
            assert isinstance(init, mx.init.Initializer)
            weight = mx.sym.var(name=name + "_weight", init=init, lr_mult=lr_mult, wd_mult=wd_mult)
        elif lr_mult != 1.0 or wd_mult != 1.0:
            weight = mx.sym.var(name=name + "_weight", lr_mult=lr_mult, wd_mult=wd_mult)
        else:
            weight = None

    return mx.sym.Convolution(data=data,
                              name=name,
                              weight=weight,
                              bias=bias,
                              num_filter=filter,
                              kernel=kernel,
                              stride=stride,
                              pad=pad,
                              dilate=dilate,
                              num_group=num_group,
                              workspace=512,
                              no_bias=no_bias)


def dwconv(data, name, filter, kernel=1, stride=1, pad=None, no_bias=True,
    init=None, lr_mult=1.0, wd_mult=1.0, weight=None, bias=None):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if pad is None:
        assert kernel[0] % 2 == 1, "Specify pad for an even kernel size for {}".format(name)
        pad = kernel[0] // 2
    if isinstance(pad, int):
        pad = (pad, pad)

    # specific initialization method
    if not isinstance(weight, mx.sym.Symbol):
        if init is not None:
            assert isinstance(init, mx.init.Initializer)
            weight = mx.sym.var(name=name + "_weight", init=init, lr_mult=lr_mult, wd_mult=wd_mult)
        elif lr_mult != 1.0 or wd_mult != 1.0:
            weight = mx.sym.var(name=name + "_weight", lr_mult=lr_mult, wd_mult=wd_mult)
        else:
            weight = None

    return mx.sym.Convolution(data=data,
                              name=name,
                              weight=weight,
                              bias=bias,
                              num_filter=filter,
                              kernel=kernel,
                              stride=stride,
                              pad=pad,
                              num_group=filter,
                              workspace=512,
                              no_bias=no_bias)


def fc(data, name, filter, flatten=True, no_bias=False, init=None,
       lr_mult=1.0, wd_mult=1.0, weight=None, bias=None):

    # specific initialization method
    if not isinstance(weight, mx.sym.Symbol):
        if init is not None:
            assert isinstance(init, mx.init.Initializer)
            weight = mx.sym.var(name=name + "_weight", init=init, lr_mult=lr_mult, wd_mult=wd_mult)
        elif lr_mult != 1.0 or wd_mult != 1.0:
            weight = mx.sym.var(name=name + "_weight", lr_mult=lr_mult, wd_mult=wd_mult)
        else:
            weight = None

    return mx.sym.FullyConnected(data=data,
                                 name=name,
                                 weight=weight,
                                 bias=bias,
                                 num_hidden=filter,
                                 no_bias=no_bias,
                                 flatten=flatten)


def fixbn(data, name, eps=1e-5+1e-10, lr_mult=1.0, wd_mult=1.0):
    return mx.sym.BatchNorm(data=data,
                            name=name,
                            eps=eps,
                            use_global_stats=True,
                            fix_gamma=False,
                            lr_mult=lr_mult,
                            wd_mult=wd_mult)


def gn(data, name, num_group=32, eps=1e-5, lr_mult=1.0, wd_mult=1.0):
    scale = mx.sym.var(name + "_scale", init=one_init(), lr_mult=lr_mult, wd_mult=wd_mult)
    bias = mx.sym.var(name + "_bias", init=zero_init(), lr_mult=lr_mult, wd_mult=wd_mult)
    return mx.sym.contrib.GroupNorm(data=data,
                                    name=name,
                                    gamma=scale,
                                    beta=bias,
                                    num_group=num_group,
                                    eps=eps)


def softmax(data, name, axis=1):
    return mx.sym.softmax(data=data, name=name, axis=axis)


def pool(data, name, kernel=3, stride=2, pad=None,
         pool_type='max', pooling_convention='valid',
         global_pool=False):
    if global_pool:
        assert pad is None
        sym = mx.sym.Pooling(data, name=name, kernel=(1, 1), pool_type="avg", global_pool=True)
        return sym
    else:
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if pad is None:
            assert kernel[0] % 2 == 1, 'Specify pad for an even kernel size'
            pad = kernel[0] // 2
        if isinstance(pad, int):
            pad = (pad, pad)
        return mx.sym.Pooling(data,
                              name=name,
                              kernel=kernel,
                              stride=stride,
                              pad=pad,
                              pool_type=pool_type,
                              pooling_convention=pooling_convention,
                              global_pool=False)


def max_pool(data, name, kernel=2, stride=2, pad=None, pooling_convention='valid'):
    return pool(data, name, kernel, stride, pad, "max", pooling_convention, False)


def avg_pool(data, name, kernel=2, stride=2, pad=None, pooling_convention='valid'):
    return pool(data, name, kernel, stride, pad, "avg", pooling_convention, False)


def global_avg_pool(data, name):
    return pool(data, name, None, None, None, None, None, True)


def upsample_bilinear(data, name, scale, filter):
    if scale == 1:
        return data
    return mx.sym.UpSampling(data=data,
                             name=name,
                             lr_mult=0,
                             wd_mult=0,
                             scale=int(scale),
                             num_filter=filter,
                             sample_type='bilinear',
                             num_args=2,
                             workspace=512)


def dense_softmax_cross_entropy_with_ignore(data, name, ignore_label, batch_size_per_gpu):
    """
    Dense softmax cross entropy with ignore computes softmax activation along axis=1, and
    normalize gradient with number of elements not ignored in NHW
    :param data:
    :param name:
    :param ignore_label:
    :param batch_size_per_gpu:
    :return:
    """
    return mx.sym.SoftmaxOutput(data=data,
                                normalization='valid',
                                multi_output=True,
                                use_ignore=True,
                                ignore_label=ignore_label,
                                grad_scale=batch_size_per_gpu,
                                name=name)


def split_channel(data, num_output, name):
    return mx.sym.split(data=data,
                        name=name,
                        num_outputs=num_output,
                        axis=1,
                        squeeze_axis=False)


def coin(prob, name):
    """
    return 1 at given probability, 0 otherwise.
    :param prob:
    :param name:
    :return:
    """
    one = mx.sym.ones(1, name=name + "_one")
    zero = mx.sym.zeros(1, name=name + "_zero")
    flip = mx.sym.random_uniform(0, 1, shape=(1, ), name=name + "_{}_flip".format(prob))
    result = mx.sym.where(flip < prob, one, zero, name=name + "_result")
    return result


def bilinear_resize(data, name, scale, filter):
    if scale == 1:
        return data

    return mx.sym.UpSampling(
        data,
        scale=scale,
        num_filter=filter,
        sample_type='bilinear',
        name=name,
        lr_mult=0,
        wd_mult=0)


def missing(*args, **kwargs):
    raise AttributeError("Your mxnet does not support this operator!")


var = mx.sym.var

group = mx.sym.Group

proposal = mx.sym.contrib.Proposal

try:
    proposal_target = mx.sym.ProposalTarget
except AttributeError:
    print("\033[91m" + "[Warning] Your mxnet does not support ProposalTarget" + "\033[0m")

from mxnext.tvm.decode_bbox import decode_bbox as _decode_bbox
def decode_bbox(rois, bbox_pred, im_info, bbox_mean, bbox_std, class_agnostic, name):
    return _decode_bbox(mx.sym, rois, bbox_pred, im_info, bbox_mean, bbox_std, class_agnostic)

try:
    bbox_norm = mx.sym.contrib.BBoxNorm
except AttributeError:
    print("\033[91m" + "[Warning] Your mxnet does not support BBoxNorm" + "\033[0m")

try:
    focal_loss = mx.sym.contrib.FocalLoss
except AttributeError:
    print("\033[91m" + "[Warning] Your mxnet does not support FocalLoss" + "\033[0m")

l2norm = mx.sym.L2Normalization

batch_dot = mx.sym.batch_dot

identity = mx.sym.identity

in_ = mx.sym.InstanceNorm

broadcast_like = mx.sym.broadcast_like

argmax_channel = mx.sym.argmax_channel

add = mx.sym.elemwise_add

sub = mx.sym.elemwise_sub

minus = sub

mult = mx.sym.elemwise_mul

div = mx.sym.elemwise_div

flatten = mx.sym.flatten

reshape = mx.sym.reshape

loss = mx.sym.MakeLoss

smooth_l1 = mx.sym.smooth_l1

block_grad = mx.sym.stop_gradient

stop_grad = mx.sym.stop_gradient

softmax_output = mx.sym.SoftmaxOutput

add_n = mx.sym.add_n

abs = mx.sym.abs

make_loss = mx.sym.MakeLoss

transpose = mx.sym.transpose

to_fp16 = lambda data, name: mx.sym.cast(data, dtype=np.float16, name=name)

to_fp32 = lambda data, name: mx.sym.cast(data, dtype=np.float32, name=name)


# combination
def convrelu(data, name, filter, kernel=1, stride=1, pad=None, dilate=1, num_group=1, no_bias=True, init=None,
             lr_mult=1.0, wd_mult=1.0, weight=None, bias=None):
    d1 = conv(data, name, filter, kernel, stride, pad, dilate, num_group, no_bias, init, lr_mult, wd_mult, weight, bias)
    d2 = relu(d1, name + "_relu")
    return d2


def convgnrelu(data, name, filter, kernel=1, stride=1, pad=None, dilate=1, num_group=1, no_bias=True, init=None,
               conv_lr_mult=1.0, conv_wd_mult=1.0, gn_lr_mult=1.0, gn_wd_mult=1.0, weight=None, bias=None):
    d1 = conv(data, name, filter, kernel, stride, pad, dilate, num_group, no_bias, init, conv_lr_mult, conv_wd_mult, weight, bias)
    d2 = gn(d1, name=name + "_gn", lr_mult=gn_lr_mult, wd_mult=gn_wd_mult)
    d3 = relu(d2, name + "_relu")
    return d3


def convnormrelu(norm, data, name, filter, kernel=1, stride=1, pad=None, dilate=1, num_group=1, no_bias=True, init=None,
                 conv_lr_mult=1.0, conv_wd_mult=1.0, norm_lr_mult=1.0, norm_wd_mult=1.0, weight=None, bias=None):
    d1 = conv(data, name, filter, kernel, stride, pad, dilate, num_group, no_bias, init, conv_lr_mult, conv_wd_mult, weight, bias)
    # _bn will be replaced by _in, _gn or _ibn accordingly
    d2 = norm(d1, name=name + "_bn", lr_mult=norm_lr_mult, wd_mult=norm_wd_mult)
    d3 = relu(d2, name + "_relu")
    return d3


def reluconv(data, name, filter, kernel=1, stride=1, pad=None, dilate=1, num_group=1, no_bias=True, init=None,
             lr_mult=1.0, wd_mult=1.0, weight=None, bias=None):
    d1 = relu(data, name=name + "_relu")
    d2 = conv(d1, name, filter, kernel, stride, pad, dilate, num_group, no_bias, init, lr_mult, wd_mult, weight, bias)
    return d2

def merge_sum(sum_list, name):
    return mx.sym.ElementWiseSum(*sum_list, name=name + '_sum')

def reluconvbn(data, filters, init, norm, name, prefix, kernel=3, pad=1, stride=1):
    data = mx.sym.Activation(data, name=name + '_relu', act_type='relu')
    weight = mx.sym.var(name=prefix + name + "_weight", init=init)
    bias = mx.sym.var(name=prefix + name + "_bias")
    data = mx.sym.Convolution(data, name=name, weight=weight, bias=bias, \
                num_filter=filters, kernel=(kernel, kernel), pad=(pad, pad), stride=(stride, stride))
    data = norm(data, name=name+'_bn_neck')
    return data
