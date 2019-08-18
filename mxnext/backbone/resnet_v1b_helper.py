from __future__ import division

from ..simple import reluconv, conv, pool, relu, add, whiten, var, fixbn, to_fp16
from ..complicate import normalizer_factory


depth_config = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3)
}

def resnet_unit(input, name, filter, stride, dilate, proj, norm):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", filter=filter // 4, kernel=3, stride=stride, dilate=dilate)
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")

def resnet_stage(data, name, num_block, filter, stride, dilate, norm):
    s, d = stride, dilate

    data = resnet_unit(data, "{}_unit1".format(name), filter, s, d, True, norm)
    for i in range(2, num_block + 1):
        data = resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm)

    return data

def resnet_c1(data, norm):
    # C1
    data = conv(data, filter=64, kernel=7, stride=2, name="conv0")
    data = norm(data, name='bn0')
    data = relu(data, name='relu0')

    data = pool(data, name="pool0", kernel=3, stride=2, pool_type='max')

    return data

def resnet_c2(data, num_block, stride, dilate, norm):
    return resnet_stage(data, "stage1", num_block, 256, stride, dilate, norm)

def resnet_c3(data, num_block, stride, dilate, norm):
    return resnet_stage(data, "stage2", num_block, 512, stride, dilate, norm)

def resnet_c4(data, num_block, stride, dilate, norm):
    return resnet_stage(data, "stage3", num_block, 1024, stride, dilate, norm)

def resnet_c5(data, num_block, stride, dilate, norm):
    return resnet_stage(data, "stage4", num_block, 2048, stride, dilate, norm)
