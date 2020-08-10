from __future__ import division

from ..simple import reluconv, conv, pool, relu, add, whiten, var, fixbn, to_fp16
from ..complicate import normalizer_factory


depth_config = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3)
}


def resnext_unit(data, name, filter, stride, dilate, proj, group, channel_per_group, norm):
    conv1 = conv(data=data, name=name + "_conv1", filter=group * channel_per_group * filter // 256)
    bn1 = norm(data=conv1, name=name + "_bn1")
    relu1 = relu(bn1)

    conv2 = conv(data=relu1, name=name + "_conv2", filter=group * channel_per_group * filter // 256, kernel=3, num_group=group, stride=stride, dilate=dilate)
    bn2 = norm(data=conv2, name=name + "_bn2")
    relu2 = relu(bn2)

    conv3 = conv(data=relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(data=conv3, name=name + "_bn3")

    if proj:
        shortcut_conv = conv(data, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(data=shortcut_conv, name=name + "_sc_bn")
    else:
        shortcut = data

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def resnext_stage(data, name, num_block, filter, stride, dilate, group, channel_per_group, norm):
    s, d = stride, dilate

    data = resnext_unit(data, "{}_unit1".format(name), filter, s, d, True, group, channel_per_group, norm)
    for i in range(2, num_block + 1):
        data = resnext_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, group, channel_per_group, norm)

    return data

def resnext_c1(data, norm):
    data = conv(data, filter=64, kernel=7, stride=2, name="conv0")
    data = norm(data, name='bn0')
    data = relu(data, name='relu0')

    data = pool(data, name="pool0", kernel=3, stride=2, pool_type='max')

    return data

def resnext_c2(data, num_block, stride, dilate, group, channel_per_group, norm):
    return resnext_stage(data, "stage1", num_block, 256, stride, dilate, group, channel_per_group, norm)

def resnext_c3(data, num_block, stride, dilate, group, channel_per_group, norm):
    return resnext_stage(data, "stage2", num_block, 512, stride, dilate, group, channel_per_group, norm)

def resnext_c4(data, num_block, stride, dilate, group, channel_per_group, norm):
    return resnext_stage(data, "stage3", num_block, 1024, stride, dilate, group, channel_per_group, norm)

def resnext_c5(data, num_block, stride, dilate, group, channel_per_group, norm):
    return resnext_stage(data, "stage4", num_block, 2048, stride, dilate, group, channel_per_group, norm)

