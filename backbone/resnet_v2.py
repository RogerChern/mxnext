from __future__ import division


from ..simple import reluconv, conv, pool, relu, add, whiten, var
from ..complicate import normalizer_factory


__all__ = ["Builder"]


class Builder(object):
    depth_config = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
            200: (3, 24, 36, 3)
    }

    @staticmethod
    def resnet_unit(data, name, filter, stride, dilate, proj, norm_type, norm_mom, ndev):
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

        bn1 = norm(data=data, name=name + "_bn1")
        relu1 = relu(bn1)
        conv1 = conv(relu1, name=name + "_conv1", filter=filter // 4)

        bn2 = norm(data=conv1, name=name + "_bn2")
        conv2 = reluconv(bn2, name=name + "_conv2", filter=filter // 4, kernel=3, stride=stride, dilate=dilate)

        bn3 = norm(data=conv2, name=name + "_bn3")
        conv3 = reluconv(bn3, name=name + "_conv3", filter=filter)

        if proj:
            shortcut = conv(relu1, name=name + "_sc", filter=filter, stride=stride)
        else:
            shortcut = data

        return add(conv3, shortcut, name=name + "_plus")

    @staticmethod
    def resnet_stage(data, name, num_block, filter, stride, dilate, norm_type, norm_mom, ndev):
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

        data = Builder.resnet_unit(data, "{}_unit1".format(name), filter, s, d, True, norm_type, norm_mom, ndev)
        for i in range(2, num_block + 1):
            data = Builder.resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm_type, norm_mom, ndev)

        return data

    @staticmethod
    def resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev):
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

    @staticmethod
    def resnet_c2(data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return Builder.resnet_stage(data, "stage1", num_block, 256, stride, dilate, norm_type, norm_mom, ndev)

    @staticmethod
    def resnet_c3(data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return Builder.resnet_stage(data, "stage2", num_block, 512, stride, dilate, norm_type, norm_mom, ndev)

    @staticmethod
    def resnet_c4(data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return Builder.resnet_stage(data, "stage3", num_block, 1024, stride, dilate, norm_type, norm_mom, ndev)

    @staticmethod
    def resnet_c5(data, num_block, stride, dilate, norm_type, norm_mom, ndev):
        return Builder.resnet_stage(data, "stage4", num_block, 2048, stride, dilate, norm_type, norm_mom, ndev)

    @staticmethod
    def resnet_c4_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None):
        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = Builder.depth_config[depth]

        data = var("data")
        c1 = Builder.resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
        c2 = Builder.resnet_c2(c1, num_c2_unit, 1, 1, norm_type, norm_mom, ndev)
        c3 = Builder.resnet_c3(c2, num_c3_unit, 2, 1, norm_type, norm_mom, ndev)
        c4 = Builder.resnet_c4(c3, num_c4_unit, 2, 1, norm_type, norm_mom, ndev)

        return c4

    @staticmethod
    def resnet_c5_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None):
        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = Builder.depth_config[depth]

        data = var("data")
        c1 = Builder.resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
        c2 = Builder.resnet_c2(c1, num_c2_unit, 1, 1, norm_type, norm_mom, ndev)
        c3 = Builder.resnet_c3(c2, num_c3_unit, 2, 1, norm_type, norm_mom, ndev)
        c4 = Builder.resnet_c4(c3, num_c4_unit, 2, 1, norm_type, norm_mom, ndev)
        c5 = Builder.resnet_c5(c4, num_c5_unit, 1, 2, norm_type, norm_mom, ndev)

        return c5

    @staticmethod
    def resnet_c4c5_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None):
        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = Builder.depth_config[depth]

        data = var("data")
        c1 = Builder.resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
        c2 = Builder.resnet_c2(c1, num_c2_unit, 1, 1, norm_type, norm_mom, ndev)
        c3 = Builder.resnet_c3(c2, num_c3_unit, 2, 1, norm_type, norm_mom, ndev)
        c4 = Builder.resnet_c4(c3, num_c4_unit, 2, 1, norm_type, norm_mom, ndev)
        c5 = Builder.resnet_c5(c4, num_c5_unit, 1, 2, norm_type, norm_mom, ndev)

        return c4, c5

    @staticmethod
    def resnet_fpn_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None):
        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = Builder.depth_config[depth]

        data = var("data")
        c1 = Builder.resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
        c2 = Builder.resnet_c2(c1, num_c2_unit, 1, 1, norm_type, norm_mom, ndev)
        c3 = Builder.resnet_c3(c2, num_c3_unit, 2, 1, norm_type, norm_mom, ndev)
        c4 = Builder.resnet_c4(c3, num_c4_unit, 2, 1, norm_type, norm_mom, ndev)
        c5 = Builder.resnet_c5(c4, num_c5_unit, 2, 1, norm_type, norm_mom, ndev)
        p6 = pool(c5, name="c5_2x_downsampled", kernel=1)

        return c2, c3, c4, c5, p6

    def __getattr__(self, name):
        """
        Allow user specifing backbone as a string in the config file
        backbone comes in 4 parts of format variant_network_last_norm-type, e.g. tusimple_resnet50_c4_fixbn
        - variants can be tusimple or tornadomeet
        - networks can be resnetXXv2
        - last can be c4, c5 or fpn
        - norm type can be fixbn, localbn, syncbn or gn
        :param name: backbone type
        :return: symbol or a tuple of symbols(FPN)
        """
        print("get symbol {}".format(name))

        parts = name.split("_")
        assert len(parts) == 4
        variant, network, last_layer, normalizer = parts

        # parse variant
        if variant == "tornadomeet":
            use_bn_preprocess = True
            use_3x3_conv0 = False
        elif variant == "tusimple":
            use_bn_preprocess = False
            use_3x3_conv0 = True
        else:
            raise KeyError("Unknown backbone variant {}".format(name))

        # parse network
        if network == "resnet50":
            depth = 50
        elif network == "resnet101":
            depth = 101
        elif network == "resnet152":
            depth = 152
        elif network == "resnet200":
            depth = 200
        else:
            raise KeyError("Unknown backbone network {}".format(name))

        # parse last layer
        if last_layer == "c4":
            factory = Builder.resnet_c4_factory
        elif last_layer == "c5":
            factory = Builder.resnet_c5_factory
        elif last_layer == "c4c5":
            factory = Builder.resnet_c4c5_factory
        elif last_layer == "fpn":
            factory = Builder.resnet_fpn_factory
        else:
            raise KeyError("Unknown backbone last layer {}".format(name))

        # parse normalizer
        if normalizer == "syncbn":
            return lambda n: factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type="sync", ndev=n)
        elif normalizer == "fixbn":
            norm_type = "fix"
        elif normalizer == "gn":
            norm_type = "gn"
        else:
            raise KeyError("Unknown backbone normalizer {}".format(name))

        return lambda: factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type=norm_type)


# TODO: hook import with ResNetV2Builder
# import sys
# sys.modules[__name__] = ResNetV2Builder()


if __name__ == "__main__":
    #############################################################
    # python -m mxnext.backbone.resnet_v2
    #############################################################

    h = Builder()
    sym = getattr(h, "tornadomeet_resnet50_c5_syncbn")(8)
    import mxnet as mx
    mx.viz.print_summary(sym)
