from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from .simple import concat, split_channel


import mxnet as mx


__all__ = ["normalizer_factory"]

bn_count = [0]


def normalizer_factory(type="local", ndev=None, eps=1e-5 + 1e-10, mom=0.9):
    """
    :param type: one of "fix", "local", "sync"
    :param ndev:
    :param eps:
    :param mom: momentum of moving mean and moving variance
    :return: a wrapper with signature, bn(data, name)
    """
    if type == "local":
        def local_bn(data, name=None, momentum=mom):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=False,
                                    momentum=momentum,
                                    eps=eps)
        return local_bn

    elif type == "fix":
        def fix_bn(data, name=None):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=True,
                                    eps=eps)
        return fix_bn

    elif type == "sync":
        assert ndev is not None, "Specify ndev for sync bn"

        def sync_bn(data, name=None, momentum=mom):
            bn_count[0] = bn_count[0] + 1
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.contrib.SyncBatchNorm(data=data,
                                                name=name,
                                                fix_gamma=False,
                                                use_global_stats=False,
                                                momentum=momentum,
                                                eps=eps,
                                                wd_mult=0,
                                                ndev=ndev,
                                                key=str(bn_count[0]))
        return sync_bn

    elif type == "in":
        def in_(data, name=None):
            if name is None:
                prev_name = data.name
                name = prev_name + "_in"
            name = name.replace("_bn", "_in")
            return mx.sym.InstanceNorm(data=data,
                                       name=name,
                                       eps=eps)
        return in_

    elif type == "gn":
        def gn(data, name=None):
            if name is None:
                prev_name = data.name
                name = prev_name + "_gn"
            name = name.replace("_bn", "_gn")
            return mx.sym.contrib.GroupNorm(data=data,
                                            name=name,
                                            eps=eps,
                                            num_group=32)
        return gn

    elif type == "ibn":
        def ibn(data, name=None, momentum=mom):
            bn_count[0] = bn_count[0] + 1
            if name is None:
                prev_name = data.name
                name = prev_name + "_ibn"
            name = name.replace("_bn", "_ibn")
            split1, split2 = split_channel(data, 2, name + "_split")
            _in = mx.sym.InstanceNorm(data=split1, name=name + "_in", eps=eps)
            _bn = mx.sym.contrib.SyncBatchNorm(data=split2,
                                               name=name + "_bn",
                                               fix_gamma=False,
                                               use_global_stats=False,
                                               momentum=momentum,
                                               eps=eps,
                                               wd_mult=0,
                                               ndev=ndev,
                                               key=str(bn_count[0]))
            _ibn = concat([_in, _bn], name=name + "_concat")
            return _ibn
        return ibn

    else:
        raise KeyError("Unknown batchnorm type {}".format(type))
