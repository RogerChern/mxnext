from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from .simple import add


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
    if type == "local" or type == "localbn":
        def local_bn(data, name=None, momentum=mom, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=False,
                                    momentum=momentum,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)
        return local_bn

    elif type == "fix" or type == "fixbn":
        def fix_bn(data, name=None, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_bn"
            return mx.sym.BatchNorm(data=data,
                                    name=name,
                                    fix_gamma=False,
                                    use_global_stats=True,
                                    eps=eps,
                                    lr_mult=lr_mult,
                                    wd_mult=wd_mult)
        return fix_bn

    elif type == "sync" or type == "syncbn":
        assert ndev is not None, "Specify ndev for sync bn"

        def sync_bn(data, name=None, momentum=mom, lr_mult=1.0, wd_mult=1.0):
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
                                                ndev=ndev,
                                                key=str(bn_count[0]),
                                                lr_mult=lr_mult,
                                                wd_mult=wd_mult)
        return sync_bn

    elif type == "in":
        def in_(data, name=None, lr_mult=1.0, wd_mult=0.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_in"
            name = name.replace("_bn", "_in")
            return mx.sym.InstanceNorm(data=data,
                                       name=name,
                                       eps=eps,
                                       lr_mult=lr_mult,
                                       wd_mult=wd_mult)
        return in_

    elif type == "gn":
        def gn(data, name=None, lr_mult=1.0, wd_mult=1.0):
            if name is None:
                prev_name = data.name
                name = prev_name + "_gn"
            name = name.replace("_bn", "_gn")
            return mx.sym.contrib.GroupNorm(data=data,
                                            name=name,
                                            eps=eps,
                                            num_group=32,
                                            lr_mult=lr_mult,
                                            wd_mult=wd_mult)
        return gn

    elif type == "ibn.1":
        def ibn(data, name=None, momentum=mom, lr_mult=1.0, wd_mult=1.0):
            bn_count[0] = bn_count[0] + 1
            if name is None:
                prev_name = data.name
                name = prev_name + "_ibn"
            name = name.replace("_bn", "_ibn")
            _in = mx.sym.InstanceNorm(data=data,
                                      name=name + "_in",
                                      eps=eps,
                                      lr_mult=lr_mult,
                                      wd_mult=wd_mult)

            _bn = mx.sym.contrib.SyncBatchNorm(data=data,
                                               name=name + "_bn",
                                               fix_gamma=False,
                                               use_global_stats=False,
                                               momentum=momentum,
                                               eps=eps,
                                               ndev=ndev,
                                               key=str(bn_count[0]),
                                               lr_mult=lr_mult,
                                               wd_mult=wd_mult)
            _ibn = add(_in, _bn, name=name + "_add")
            return _ibn
        return ibn

    else:
        raise KeyError("Unknown batchnorm type {}".format(type))
