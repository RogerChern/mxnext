import mxnet as mx


def _bbox_encode(F, ex_rois, gt_rois):
    ex_x1, ex_y1, ex_x2, ex_y2 = F.split(ex_rois, num_outputs=4, axis=-1)
    gt_x1, gt_y1, gt_x2, gt_y2 = F.split(gt_rois, num_outputs=4, axis=-1)

    ex_widths = ex_x2 - ex_x1 + 1.0
    ex_heights = ex_y2 - ex_y1 + 1.0
    ex_ctr_x = ex_x1 + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_y1 + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_x2 - gt_x1 + 1.0
    gt_heights = gt_y2 - gt_y1 + 1.0
    gt_ctr_x = gt_x1 + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_y1 + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = F.log(gt_widths / ex_widths)
    targets_dh = F.log(gt_heights / ex_heights)

    return F.concat(targets_dx, targets_dy, targets_dw, targets_dh, dim=-1)


def _rpn_target_single_scale_batch(F, features, anchors, gt_bboxes, im_infos, num_image, num_anchor,
    max_side, feature_stride, allowed_border, sample_per_image, fg_fraction, fg_thr, bg_thr):
    F = mx.ndarray

    max_side = max_side // feature_stride

    # prepare output
    anchors = F.slice_like(anchors, features, axes=(2, 3))  # (1, 1, h, w, #anchor * 4)
    anchors = anchors.reshape(-3, -2).reshape(0, 0, -1, 4)  # (1, h, w, #anchor, 4)
    anchors = F.broadcast_axis(anchors, axis=0, size=num_image)  # (n, h, w, #anchor, 4)
    cls_label_shape = F.slice_axis(anchors, axis=-1, begin=0, end=1).reshape(num_image, -1)  # (n, h * w * #anchor)
    rpn_cls_label = F.ones_like(cls_label_shape) * -1  # -1 for ignore

    # prepare anchor, gt, valid
    anchors = F.reshape(anchors, shape=(num_image, -1, 4))  # (n, h * w * #anchor, 4)
    gt_bboxes  # (n, #gt, 4)
    im_infos  # (n, 3)
    h, w, _ = F.split(im_infos, num_outputs=3, axis=-1)  # (n, 1)
    x1, y1, x2, y2 = F.split(anchors, num_outputs=4, axis=-1, squeeze_axis=True)  # (n, h * w * #anchor)
    valid = (x1 >= -allowed_border) * \
            F.broadcast_lesser(x2, w + allowed_border) * \
            (y1 >= -allowed_border) * \
            F.broadcast_lesser(y2, h + allowed_border)  # (n, h * w * #anchor)

    ######################## assgining #######################
    iou_a2gt_list = list()
    for i in range(num_image):
        anchors_this = F.slice_axis(anchors, axis=0, begin=i, end=i+1).reshape(-1, 4)
        gt_bboxes_this = F.slice_axis(gt_bboxes, axis=0, begin=i, end=i+1).reshape(-1, 4)
        iou_a2gt_this = F.contrib.box_iou(anchors_this, gt_bboxes_this, format="corner")  # (h * w * #anchor, #gt)
        iou_a2gt_list.append(iou_a2gt_this)
    iou_a2gt = F.stack(*iou_a2gt_list, axis=0)  # (n, h * w * #anchor, #gt)
    matched_gt_idx = F.argmax(iou_a2gt, axis=-1)  # (n, h * w * #anchor)
    max_iou_a = F.max(iou_a2gt, axis=-1)  # (n, h * w * #anchor)
    max_iou_gt = F.max(iou_a2gt, axis=-2) # (n, #gt)

    # choose anchors with IoU < bg_thr
    matched = max_iou_a < bg_thr
    rpn_cls_label = F.where(matched * valid, F.zeros_like(rpn_cls_label), rpn_cls_label)

    # choose anchors with max IoU to gts
    max_iou_gt = F.expand_dims(max_iou_gt, axis=1)  # (n, 1, #gt)
    matched = F.broadcast_equal(iou_a2gt, max_iou_gt) * (iou_a2gt > 0)  # (n, h * w * #anchor, #gt)
    matched = F.sum(matched, axis=-1)  # (n, h * w * #anchor)
    matched = matched > 0
    rpn_cls_label = F.where(matched * valid, F.ones_like(rpn_cls_label), rpn_cls_label)

    # choose anchors with IoU >= fg_thr
    matched = max_iou_a >= fg_thr
    rpn_cls_label = F.where(matched * valid, F.ones_like(rpn_cls_label), rpn_cls_label)

    ######################## sampling ##########################
    # rpn_cls_label is 1 for fg, 0 for bg and -1 for ignore
    ############################################################

    # simulate random sampling by add a random number [0, 0.5] to each label and sort
    randp = F.random.uniform(0, 0.5, shape=(num_image, max_side ** 2 * num_anchor))
    randp = F.slice_like(randp, rpn_cls_label)
    rpn_cls_label_p = rpn_cls_label + randp

    # filter out excessive fg samples
    fg_value = F.topk(rpn_cls_label_p, k=int(sample_per_image * fg_fraction), ret_typ='value')
    sentinel_value = F.slice_axis(fg_value, axis=-1, begin=-1, end=None)  # (n, 1)
    matched = rpn_cls_label_p >= 1
    matched = matched * F.broadcast_lesser(rpn_cls_label_p, sentinel_value)
    rpn_cls_label = F.where(matched, F.ones_like(rpn_cls_label) * -1, rpn_cls_label)

    # filter out excessive bg samples
    matched = rpn_cls_label_p < 1
    rpn_cls_label_p = F.where(matched, rpn_cls_label_p + 3, rpn_cls_label_p)
    # now [3, 3.5] for bg, [2, 2.5] for ignore and [1, 1.5] for fg
    fg_value = F.topk(rpn_cls_label_p, k=sample_per_image - int(sample_per_image * fg_fraction), ret_typ='value')
    sentinel_value = F.slice_axis(fg_value, axis=-1, begin=-1, end=None)  # (n, 1)
    matched = rpn_cls_label_p >= 3
    matched = matched * F.broadcast_lesser(rpn_cls_label_p, sentinel_value)
    # rpn_cls_label = F.where(matched, F.ones_like(rpn_cls_label) * -1, rpn_cls_label)

    # regression target
    matched = (rpn_cls_label == 1).reshape(-1)
    ones = F.ones_like(anchors).reshape(-1, 4)
    zeros = F.zeros_like(anchors).reshape(-1, 4)
    rpn_reg_weight = F.where(matched, ones, zeros)

    # get matched gt box
    matched_gt_box_list = list()
    for i in range(num_image):
        matched_gt_idx_this = F.slice_axis(matched_gt_idx, axis=0, begin=i, end=i+1).reshape(-1)
        gt_bboxes_this = F.slice_axis(gt_bboxes, axis=0, begin=i, end=i+1).reshape(-1, 4)
        matched_gt_box_list.append(F.take(gt_bboxes_this, matched_gt_idx_this))
    matched_gt_box = F.stack(*matched_gt_box_list, axis=0)
    rpn_reg_target = _bbox_encode(F, anchors, matched_gt_box)

    # reshape output
    rpn_cls_label = F.reshape(rpn_cls_label, shape=(num_image, -1, num_anchor))
    rpn_cls_label = F.reshape_like(rpn_cls_label, features, lhs_begin=1, lhs_end=2, rhs_begin=2, rhs_end=4)
    rpn_cls_label = F.transpose(rpn_cls_label, axes=(0, 3, 1, 2)).reshape(num_image, -1)

    rpn_reg_target = F.reshape(rpn_reg_target, shape=(num_image, -1, num_anchor * 4))
    rpn_reg_target = F.reshape_like(rpn_reg_target, features, lhs_begin=1, lhs_end=2, rhs_begin=2, rhs_end=4)
    rpn_reg_target = F.transpose(rpn_reg_target, axes=(0, 3, 1, 2))

    rpn_reg_weight = F.reshape(rpn_reg_weight, shape=(num_image, -1, num_anchor * 4))
    rpn_reg_weight = F.reshape_like(rpn_reg_weight, features, lhs_begin=1, lhs_end=2, rhs_begin=2, rhs_end=4)
    rpn_reg_weight = F.transpose(rpn_reg_weight, axes=(0, 3, 1, 2))

    return rpn_cls_label, rpn_reg_target, rpn_reg_weight


def test_rpn_target():
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
    anchors = mx.nd.array(all_anchor, dtype="float32")
    cls_prob = mx.nd.random_normal(0, 1, shape=[1, len(aspects) * len(scales), feat_h, feat_w])
    gt_bboxes = mx.nd.array(
        [[200, 200, 300, 300],
         [300, 300, 500, 500],
         [-1, -1, -1, -1],
         [200, 200, 300, 300],
         [400, 300, 500, 500],
         [-1, -1, -1, -1],
         ]).reshape(2, 3, 4)
    im_infos = mx.nd.array([[1200, 800, 2], [1200, 800, 2]]).reshape(2, 3)

    rpn_cls_label, rpn_reg_target, rpn_reg_weight = _rpn_target_single_scale_batch(mx.ndarray, cls_prob,
        anchors, gt_bboxes, im_infos, 2, 15, max_side, stride, 0, 256, 0.5, 0.7, 0.3)
    print(len(np.where(rpn_cls_label[1].asnumpy() == 0)[0]))
    print(np.where(rpn_cls_label[1].asnumpy() > 0))
    print(np.where(rpn_reg_weight[1].asnumpy() > 0))
    print(rpn_reg_target[1][np.where(rpn_reg_weight[1].asnumpy() > 0)])

    from core.detection_input import AnchorTarget2D
    class AnchorTarget2DParam:
        class generate:
            short = 800 // 16
            long = 1200 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5

    anchor_target = AnchorTarget2D(AnchorTarget2DParam)

    record = {"im_info": im_infos.asnumpy()[1], "gt_bbox": gt_bboxes.asnumpy()[1]}
    anchor_target.apply(record)
    print(len(np.where(record["rpn_cls_label"] == 0)[0]))
    print(np.where(record["rpn_cls_label"] > 0))
    print(np.where(record["rpn_reg_weight"] > 0))
    print(record["rpn_reg_target"][np.where(record["rpn_reg_weight"] > 0)])

if __name__ == "__main__":
    test_rpn_target()
