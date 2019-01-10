from __future__ import division


from ..simple import reluconv, conv, pool, relu, add, whiten, var, fixbn, to_fp16
from ..complicate import normalizer_factory


__all__ = ["Builder"]


class Builder(object):
    depth_config = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
            200: (3, 24, 36, 3)
    }

    @classmethod
    def resnet_unit(cls, data, name, filter, stride, dilate, proj, norm_type, norm_mom, ndev):
        """
        One resnet unit is comprised of 2 or 3 convolutions and a shortcut.
        :param data:
        :param name:
        :param filter:
        :param stride:
        :param dilate:
        :param proj:
        :param norm_type:
        :param norm_mom:
        :param ndev:
        :return:
        """
        norm = normalizer_factory(type=norm_type, ndev=ndev, mom=norm_mom)

        conv1 = conv(data, name=name + "_conv1", filter=filter // 4, stride=stride, dilate=dilate)
        bn1 = norm(data=conv1, name=name + "_bn1")
        relu1 = relu(bn1, name=name + "_relu1")

        conv2 = conv(relu1, name=name + "_conv2", filter=filter // 4, kernel=3)
        bn2 = norm(data=conv2, name=name + "_bn2")
        relu2 = relu(bn2, name=name + "_relu2")

        conv3 = conv(relu2, name=name + "_conv3", filter=filter)
        bn3 = norm(data=conv3, name=name + "_bn3")

        if proj:
            shortcut = conv(data, name=name + "_sc", filter=filter, stride=stride)
            shortcut = norm(data=shortcut, name=name + "_sc_bn")
        else:
            shortcut = data

        eltwise = add(bn3, shortcut, name=name + "_plus")

        return relu(eltwise, name=name + "_relu")

    @classmethod
    def resnet_stage(cls, data, name, num_block, filter, stride, dilate, norm_type, norm_mom, ndev):
        """
        One resnet stage is comprised of multiple resnet units. Refer to depth config for more information.
        :param data:
        :param name:
        :param num_block:
        :param filter:
        :param stride:
        :param dilate:
        :param norm_type:
        :param norm_mom:
        :param ndev:
        :return:
        """
        s, d = stride, dilate

        data = cls.resnet_unit(data, "{}_unit1".format(name), filter, s, d, True, norm_type, norm_mom, ndev)
        for i in range(2, num_block + 1):
            data = cls.resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm_type, norm_mom, ndev)

        return data

    @classmethod
    def resnet_c1(cls, data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev):
        """
        Resnet C1 is comprised of irregular initial layers.
        :param data: image symbol
        :param use_3x3_conv0: use three 3x3 convs to replace one 7x7 conv
        :param use_bn_preprocess: use batchnorm as the whitening layer, introduced by tornadomeet
        :param norm_type: normalization method of activation, could be local, fix, sync, gn, in, ibn
        :param norm_mom: normalization momentum, specific to batchnorm
        :param ndev: num of gpus for sync batchnorm
        :return: C1 symbol
        """
        # preprocess
        if use_bn_preprocess:
            data = whiten(data, name="bn_data")

        norm = normalizer_factory(type=norm_type, ndev=ndev, mom=norm_mom)

        # C1
        if use_3x3_conv0:
            data = conv(data, filter=64, kernel=3, stride=2, name="conv0_0")
            data = norm(data, name='bn0_0')
            data = relu(data, name='relu0_0')

            data = conv(data, filter=64, kernel=3, name="conv0_1")
            data = norm(data, name='bn0_1')
            data = relu(data, name='relu0_1')

            data = conv(data, filter=64, kernel=3, name="conv0_2")
            data = norm(data, name='bn0_2')
            data = relu(data, name='relu0_2')
        else:
            data = conv(data, filter=64, kernel=7, stride=2, name="conv0")
            data = norm(data, name='bn0')
            data = relu(data, name='relu0')

        data = pool(data, name="pool0", kernel=3, stride=2, pool_type='max')

        return data

    @classmethod
    def resnet_c2(cls, data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return cls.resnet_stage(data, "stage1", num_block, 256, stride, dilate, norm_type, norm_mom, ndev)

    @classmethod
    def resnet_c3(cls, data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return cls.resnet_stage(data, "stage2", num_block, 512, stride, dilate, norm_type, norm_mom, ndev)

    @classmethod
    def resnet_c4(cls, data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return cls.resnet_stage(data, "stage3", num_block, 1024, stride, dilate, norm_type, norm_mom, ndev)

    @classmethod
    def resnet_c5(cls, data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return cls.resnet_stage(data, "stage4", num_block, 2048, stride, dilate, norm_type, norm_mom, ndev)

    @classmethod
    def resnet_factory(cls, depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = cls.depth_config[depth]

        data = var("data")
        if fp16:
            data = to_fp16(data, "data_fp16")
        c1 = cls.resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
        c2 = cls.resnet_c2(c1, num_c2_unit, 1, 1, norm_type, norm_mom, ndev)
        c3 = cls.resnet_c3(c2, num_c3_unit, 2, 1, norm_type, norm_mom, ndev)
        c4 = cls.resnet_c4(c3, num_c4_unit, 2, 1, norm_type, norm_mom, ndev)
        c5 = cls.resnet_c5(c4, num_c5_unit, 2, 1, norm_type, norm_mom, ndev)

        return c1, c2, c3, c4, c5

    @classmethod
    def resnet_c4_factory(cls, depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        c1, c2, c3, c4, c5 = cls.resnet_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev, fp16)

        return c4

    @classmethod
    def resnet_c5_factory(cls, depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        c1, c2, c3, c4, c5 = cls.resnet_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev, fp16)
        c5 = fixbn(c5, "bn1")
        c5 = relu(c5)

        return c5

    @classmethod
    def resnet_c4c5_factory(cls, depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        c1, c2, c3, c4, c5 = cls.resnet_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev, fp16)
        c5 = fixbn(c5, "bn1")
        c5 = relu(c5)

        return c4, c5

    @classmethod
    def resnet_fpn_factory(cls, depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        c1, c2, c3, c4, c5 = cls.resnet_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev, fp16)

        return c2, c3, c4, c5

    def get_backbone(self, variant, depth, endpoint, normalizer, fp16):
        # parse variant
        if variant == "mxnet":
            use_bn_preprocess = True
            use_3x3_conv0 = False
        elif variant == "tusimple":
            use_bn_preprocess = False
            use_3x3_conv0 = True
        elif variant == "msra":
            use_bn_preprocess = False
            use_3x3_conv0 = False
        else:
            raise KeyError("Unknown backbone variant {}".format(variant))

        # parse endpoint
        if endpoint == "c4":
            factory = self.resnet_c4_factory
        elif endpoint == "c5":
            factory = self.resnet_c5_factory
        elif endpoint == "c4c5":
            factory = self.resnet_c4c5_factory
        elif endpoint == "fpn":
            factory = self.resnet_fpn_factory
        else:
            raise KeyError("Unknown backbone endpoint {}".format(endpoint))

        return factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type=normalizer, fp16=fp16)


# TODO: hook import with ResNetV1Builder
# import sys
# sys.modules[__name__] = ResNetV1Builder()


if __name__ == "__main__":
    #############################################################
    # python -m mxnext.backbone.resnet_v1
    #############################################################

    h = Builder()
    sym = h.get_backbone("msra", 50, "fpn", normalizer_factory(type="fixbn"), fp16=True)
    import mxnet as mx
    sym = mx.sym.Group(sym)
    mx.viz.print_summary(sym)
