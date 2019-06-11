from mxnext.tvm.decode_bbox import decode_bbox
import mxnet as mx


def proposal(
    F=mx.ndarray,
    cls_prob=None, 
    bbox_pred=None, 
    anchors=None,
    im_info=None, 
    batch_size=1,
    rpn_pre_nms_top_n=None, 
    rpn_post_nms_top_n=None, 
    threshold=None, 
    rpn_min_size=None, 
    scales=None, 
    ratios=None, 
    feature_stride=None, 
    output_score=None, 
    name=None):
    """
    cls_prob: (#img, #anchor * 2, h, w) or (#img, #anchor, h, w)
    bbox_pred: (#img, #anchor * 4, h, w)
    im_info: (#img, 3), [valid_h, valid_w, scale]

    Returns:
    proposal: (#img, #proposal, 5)
    proposal_score: (#img, #proposal, 1) if output_score == True
    """
    # constant
    num_anchor = len(scales) * len(ratios)
    max_side = 1200
    max_side = max_side // feature_stride

    # meshgrid
    xx = F.arange(0, max_side).reshape([1, max_side])
    xx = F.broadcast_axis(xx, axis=0, size=max_side).reshape([1, 1, max_side, max_side])
    xx = F.slice_like(xx, cls_prob, axes=(2, 3))
    xx = F.broadcast_axis(xx, axis=0, size=batch_size).reshape([batch_size, -1])
    yy = F.arange(0, max_side).reshape([max_side, 1])
    yy = F.broadcast_axis(yy, axis=1, size=max_side).reshape([1, 1, max_side, max_side])
    yy = F.slice_like(yy, cls_prob, axes=(2, 3))
    yy = F.broadcast_axis(yy, axis=0, size=batch_size).reshape([batch_size, -1])

    # slice anchor
    # anchors = F.var("anchor_stride%s" % feature_stride, shape=(1, 1, max_side, max_side, num_anchor * 4), dtype='float32') # (1, 1, long_side, long_side, #anchor * 4)
    anchors = F.slice_like(anchors, cls_prob, axes=(2, 3))  # (1, 1, h, w, #anchor * 4)
    anchors = F.reshape(anchors, [-3, -2])  # (1, h, w, #anchor * 4), fold first two axes
    anchors = F.broadcast_axis(anchors, axis=0, size=batch_size)  # (#img, h, w, #anchor * 4)
    anchors = anchors.reshape([0, -1, 4])  # (#img, h * w * #anchor, 4)

    # argsort
    cls_prob = F.slice_axis(cls_prob, axis=1, begin=-num_anchor, end=None)
    cls_prob = F.transpose(cls_prob, axes=[0, 2, 3, 1])  # (#img, h, w, #anchor)
    cls_prob = F.reshape(cls_prob, shape=[-1, num_anchor])
    cls_prob = F.where(F.broadcast_lesser(xx, F.floor(im_info.slice_axis(axis=1, begin=1, end=2) / feature_stride)).reshape(-1), cls_prob, -F.ones_like(cls_prob))
    cls_prob = F.where(F.broadcast_lesser(yy, F.floor(im_info.slice_axis(axis=1, begin=0, end=1) / feature_stride)).reshape(-1), cls_prob, -F.ones_like(cls_prob))
    cls_prob = F.reshape(cls_prob, shape=[batch_size, -1])  # (#img, h * w * #anchor)
    bbox_pred = F.transpose(bbox_pred, axes=[0, 2, 3, 1])  # (#img, h, w, #anchor * 4)
    bbox_pred = F.reshape(bbox_pred, shape=[0, 0, 0, -4, -1, 4]) # (#img, h, w, #anchor, 4)
    bbox_pred = F.reshape(bbox_pred, shape=[0, -1, 4])  # (#img, h * w * #anchor, 4)
    argsort_cls_prob = F.argsort(cls_prob, axis=-1, is_ascend=False)
    argsort_cls_prob = F.slice_axis(argsort_cls_prob, axis=-1, begin=0, end=rpn_pre_nms_top_n)
    arange_index = F.arange(0, batch_size).reshape([batch_size, 1])
    arange_index = F.broadcast_axis(arange_index, axis=1, size=rpn_pre_nms_top_n).reshape(-1)
    top_indexes = F.stack(arange_index, argsort_cls_prob.reshape(-1))
    sort_cls_prob = F.gather_nd(cls_prob, top_indexes).reshape([-4, -1, rpn_pre_nms_top_n])  # (#img, #proposal)
    sort_bbox_pred = F.gather_nd(bbox_pred, top_indexes).reshape([-4, -1, rpn_pre_nms_top_n, 4])  # (#img, #proposal, 4)
    sort_anchor = F.gather_nd(anchors, top_indexes).reshape([-4, -1, rpn_pre_nms_top_n, 4])  # (#img, #proposal, 4)

    # decode
    bbox = decode_bbox(F, sort_anchor, sort_bbox_pred, im_info, 
        (0, 0, 0, 0), (1, 1, 1, 1), True)

    # nms
    score_bbox = F.concat(F.expand_dims(sort_cls_prob, axis=-1), bbox, dim=-1)
    bbox = F.contrib.box_nms(score_bbox, overlap_thresh=threshold, coord_start=1, score_index=0)
    bbox_coord = F.slice_axis(bbox, axis=-1, begin=-4, end=None)
    bbox_coord = F.slice_axis(bbox_coord, axis=1, begin=0, end=rpn_post_nms_top_n)
    bbox_pad = F.broadcast_axis(F.slice_axis(bbox_coord, axis=1, begin=0, end=1), axis=1, size=rpn_post_nms_top_n)
    bbox_coord = F.where(bbox_coord >= 0, bbox_coord, bbox_pad)

    roi_index = F.arange(0, batch_size).reshape([batch_size, 1])
    roi_index = F.broadcast_axis(roi_index,axis=1, size=rpn_post_nms_top_n)
    roi_index = F.reshape(roi_index, [batch_size, rpn_post_nms_top_n, 1])
    bbox_out = F.concat(roi_index, bbox_coord, dim=-1)
    bbox_out = F.reshape(bbox_out, [-3, -2])

    return bbox_out, mx.sym.ones(1)


def test_meshgrid():
    h = 7
    w = 10

    shift_x = mx.nd.arange(0, w, repeat=h)
    grid_x = shift_x.reshape([w, h]).transpose()
    shift_y = mx.nd.arange(0, h, repeat=w)
    grid_y = shift_y.reshape([h, w])
    grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
    print(grid_x)
    print(grid_y)

    import numpy as np
    shift_x = np.arange(0, w, dtype=np.float32)
    shift_y = np.arange(0, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(shift_x, shift_y)
    grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
    print(grid_x)
    print(grid_y)


def test_proposal():
    import numpy as np

    # anchor generation
    stride = 16
    aspects = (0.5, 1.0, 2.0)
    scales = (2, 4, 8, 16, 32)
    max_side = 1200
    feat_h = 75
    feat_w = 50

    base_anchor = np.array([0, 0, stride - 1, stride - 1])
    w = base_anchor[2] - base_anchor[0] + 1
    h = base_anchor[3] - base_anchor[1] + 1
    x_ctr = base_anchor[0] + 0.5 * (w - 1)
    y_ctr = base_anchor[1] + 0.5 * (h - 1)
    w_ratios = np.round(np.sqrt(w * h / aspects))
    h_ratios = np.round(w_ratios * aspects)
    ws = (np.outer(w_ratios, scales)).reshape(-1)
    hs = (np.outer(h_ratios, scales)).reshape(-1)
    base_anchor = np.stack(
        [x_ctr - 0.5 * (ws - 1),
        y_ctr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1),
        y_ctr + 0.5 * (hs - 1)],
        axis=1)

    shift_x = np.arange(0, max_side // stride, dtype=np.float32) * stride
    shift_y = np.arange(0, max_side // stride, dtype=np.float32) * stride
    grid_x, grid_y = np.meshgrid(shift_x, shift_y)
    grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
    grid = np.stack([grid_x, grid_y, grid_x, grid_y], axis=1)
    all_anchor = grid[:, None, :] + base_anchor[None, :, :]
    all_anchor = all_anchor.reshape(1, 1, max_side // stride, max_side // stride, -1)

    cls_prob = mx.nd.random_uniform(0, 1, shape=[1, len(aspects) * len(scales), feat_h, feat_w])
    cls_prob = mx.nd.concat(mx.nd.zeros_like(cls_prob), cls_prob, dim=1)
    bbox_pred = mx.nd.random_normal(0, 0.1, shape=[1, len(aspects) * len(scales) * 4, feat_h, feat_w])
    im_info = mx.nd.array([[1101, 701, 1.0]])
    anchors = mx.nd.array(all_anchor)

    bbox1 = mx.nd.contrib.Proposal(
        cls_prob, bbox_pred, im_info, 6000, 1000, 0.7, 0, scales, aspects, stride, False, False
    )
    bbox2 = proposal(mx.ndarray, cls_prob, bbox_pred, anchors, im_info, 1, 6000, 1000, 0.7, 0, scales, aspects, stride)

    # print(bbox1 - bbox2[0])
    print(np.where((bbox1 - bbox2[0]).asnumpy().sum(axis=1) != 0)[0][:100])
    # print(bbox1[:10])
    print(bbox1[957:957+10])
    print(bbox2[0][957:957+10])

if __name__ == "__main__":
    mx.random.seed(123)
    
    test_proposal()
