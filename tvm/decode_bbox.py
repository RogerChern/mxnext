def _corner_to_center(F, boxes):
    boxes = F.slice_axis(boxes, axis=-1, begin=-4, end=None)
    xmin, ymin, xmax, ymax = F.split(boxes, axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) * 0.5
    y = ymin + (height - 1) * 0.5
    return x, y, width, height

def decode_bbox(F, anchors, deltas, im_infos, means, stds, class_agnostic):
    """
    anchors: (#img, #roi, #cls * 4)
    deltas: (#img, #roi, #cls * 4)
    im_infos: (#img, 3), [h, w, scale]
    means: (4, ), [x, y, h, w]
    stds: (4, ), [x, y, h, w]
    class_agnostic: bool
    """

    # add roi axis, layout (img, roi, coord)
    im_infos = F.expand_dims(im_infos, axis=1)

    if class_agnostic:
        # TODO: class_agnostic should predict only 1 class
        # class_agnostic predicts 2 classes
        deltas = F.slice_axis(deltas, axis=-1, begin=4, end=None)
    if not class_agnostic:
        # add class axis, layout (img, roi, cls, coord)
        deltas = F.reshape(deltas, [0, 0, -4, -1, 4])  # TODO: extend to multiple anchors
        anchors = F.expand_dims(anchors, axis=-2)
        im_infos = F.expand_dims(im_infos, axis=-2)

    ax, ay, aw, ah = _corner_to_center(F, anchors)  # anchor
    im_infos = F.split(im_infos, axis=-1, num_outputs=3)
    dx, dy, dw, dh = F.split(deltas, axis=-1, num_outputs=4)

    # delta
    dx = dx * stds[0] + means[0]
    dy = dy * stds[1] + means[1]
    dw = dw * stds[2] + means[2]
    dh = dh * stds[3] + means[3]

    # prediction
    px = F.broadcast_add(F.broadcast_mul(dx, aw), ax)
    py = F.broadcast_add(F.broadcast_mul(dy, ah), ay)
    pw = F.broadcast_mul(F.exp(dw), aw)
    ph = F.broadcast_mul(F.exp(dh), ah)

    x1 = px - 0.5 * (pw - 1.0)
    y1 = py - 0.5 * (ph - 1.0)
    x2 = px + 0.5 * (pw - 1.0)
    y2 = py + 0.5 * (ph - 1.0)

    x1 = F.maximum(F.broadcast_minimum(x1, im_infos[1] - 1.0), 0)
    y1 = F.maximum(F.broadcast_minimum(y1, im_infos[0] - 1.0), 0)
    x2 = F.maximum(F.broadcast_minimum(x2, im_infos[1] - 1.0), 0)
    y2 = F.maximum(F.broadcast_minimum(y2, im_infos[0] - 1.0), 0)

    out = F.concat(x1, y1, x2, y2, dim=-1)
    if not class_agnostic:
        out = F.reshape(out, [0, 0, -3, -2])
    return out


if __name__ == "__main__":
    import mxnet as mx

    anchors = mx.nd.array([10, 20, 55, 77, 10, 20, 50, 77]).reshape([2, 1, 4])
    im_infos = mx.nd.array([76, 72, 1, 76, 76, 1]).reshape([2, 3])
    deltas = mx.nd.array([0, 0, 0, 0.1, 0.3, 0.1, 0.2, 0.4, 0.5, 0.1, 0.2, 0.1, 0, 0, 0, 0, 0.3, 0.1, 0.2, 0.4, 0.5, 0.1, 0.2, 0.1]).reshape([2, 1, 12])
    means = (0.04, 0.01, 0.03, 0.02)
    stds = (0.5, 0.1, 0.1, 0.5)

    o1 = mx.nd.contrib.DecodeBBox(
        rois=anchors,
        bbox_pred=deltas,
        im_info=im_infos,
        bbox_mean=means,
        bbox_std=stds,
        class_agnostic=False
    )
    o2 = decode_bbox(
        mx.ndarray,
        anchors,
        deltas,
        im_infos,
        means,
        stds,
        False
    )

    print(o1)
    print(o2)
    print(o1 - o2)