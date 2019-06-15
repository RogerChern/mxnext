import numpy as np
import mxnet as mx

def fpn_roi_assign(F=mx.ndarray, rois=None, rcnn_stride=None, 
    roi_canonical_scale=None, roi_canonical_level=None):
    ############ constant #############
    scale0 = roi_canonical_scale
    lvl0 = roi_canonical_level
    k_min = np.log2(min(rcnn_stride))
    k_max = np.log2(max(rcnn_stride))

    x1, y1, x2, y2 = F.split(rois, num_outputs=4, axis=-1)
    rois_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    rois_scale = F.sqrt(rois_area)
    target_lvls = F.floor(lvl0 + F.log2(rois_scale / scale0 + 1e-6))
    target_lvls = F.clip(target_lvls, k_min, k_max)
    target_stride = F.pow(2, target_lvls).astype('uint8')

    outputs = []
    for i, s in enumerate(rcnn_stride):
        lvl_rois = F.zeros_like(rois)
        lvl_inds = (target_stride == s).astype('float32')
        lvl_inds = F.broadcast_like(lvl_inds, lvl_rois)
        lvl_rois = F.where(lvl_inds, rois, lvl_rois)
        outputs.append(lvl_rois)
    
    return outputs
