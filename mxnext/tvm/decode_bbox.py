import mxnet as mx


def _corner_to_center(F, boxes):
    boxes = F.slice_axis(boxes, axis=-1, begin=-4, end=None)
    xmin, ymin, xmax, ymax = F.split(boxes, axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) * 0.5
    y = ymin + (height - 1) * 0.5
    return x, y, width, height


def _corner_to_corner(F, boxes):
    boxes = F.slice_axis(boxes, axis=-1, begin=-4, end=None)
    xmin, ymin, xmax, ymax = F.split(boxes, axis=-1, num_outputs=4)
    return xmin, ymin, xmax, ymax


def _bbox_transform_xywh(F, anchors, deltas, means, stds):
    ax, ay, aw, ah = _corner_to_center(F, anchors)  # anchor
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
    return x1, y1, x2, y2


def _bbox_transform_xyxy(F, anchors, deltas, means, stds):
    ax1, ay1, ax2, ay2 = _corner_to_corner(F, anchors)
    aw = ax2 - ax1 + 1
    ah = ay2 - ay1 + 1
    dx1, dy1, dx2, dy2 = F.split(deltas, axis=-1, num_outputs=4)

    # delta
    dx1 = dx1 * stds[0] + means[0]
    dy1 = dy1 * stds[1] + means[1]
    dx2 = dx2 * stds[2] + means[2]
    dy2 = dy2 * stds[3] + means[3]

    # prediction
    x1 = F.broadcast_add(F.broadcast_mul(dx1, aw), ax1)
    y1 = F.broadcast_add(F.broadcast_mul(dy1, ah), ay1)
    x2 = F.broadcast_add(F.broadcast_mul(dx2, aw), ax2)
    y2 = F.broadcast_add(F.broadcast_mul(dy2, ah), ay2)
    return x1, y1, x2, y2


def decode_bbox(F, anchors, deltas, im_infos, means, stds, class_agnostic, bbox_decode_type='xywh'):
    """
    anchors: (#img, #roi, #cls * 4)
    deltas: (#img, #roi, #cls * 4)
    im_infos: (#img, 3), [h, w, scale]
    means: (4, ), [x, y, h, w]
    stds: (4, ), [x, y, h, w]
    class_agnostic: bool

    Returns:
    bbox: (#img, #roi, 4), [x1, y1, x2, y2]
    """
    with mx.name.Prefix("decode_bbox: "):
        if class_agnostic:
            # TODO: class_agnostic should predict only 1 class
            # class_agnostic predicts 2 classes
            deltas = F.slice_axis(deltas, axis=-1, begin=-4, end=None)
        if not class_agnostic:
            # add class axis, layout (img, roi, cls, coord)
            deltas = F.reshape(deltas, [0, 0, -4, -1, 4])  # TODO: extend to multiple anchors
            anchors = F.expand_dims(anchors, axis=-2)
        if bbox_decode_type == 'xywh':
            x1, y1, x2, y2 = _bbox_transform_xywh(F, anchors, deltas, means, stds)
        elif bbox_decode_type == 'xyxy':
            x1, y1, x2, y2 = _bbox_transform_xyxy(F, anchors, deltas, means, stds)
        else:
            raise NotImplementedError("decode_bbox only supports xywh or xyxy bbox_decode_type")

        if im_infos is not None:
            # add roi axis, layout (img, roi, coord)
            im_infos = F.expand_dims(im_infos, axis=1)
            if not class_agnostic:
                # add class axis, layout (img, roi, cls, coord)
                im_infos = F.expand_dims(im_infos, axis=-2)
            im_infos = F.split(im_infos, axis=-1, num_outputs=3)
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
    agnostic_deltas = mx.nd.array([0, 0, 0, 0.1, 0.3, 0.1, 0.2, 0.4, 0, 0, 0, 0, 0.3, 0.1, 0.2, 0.4]).reshape([2, 1, 8])
    means = (0.04, 0.01, 0.03, 0.02)
    stds = (0.5, 0.1, 0.1, 0.5)

    for class_agnostic in [True, False]:
        for bbox_decode_type in ['xyxy', 'xywh']:
            print("Test class_agnostic {}, bbox_decode_type {}".format(class_agnostic, bbox_decode_type))
            o1 = mx.nd.contrib.DecodeBBox(
                rois=anchors,
                bbox_pred=agnostic_deltas if class_agnostic else deltas,
                im_info=im_infos,
                bbox_mean=means,
                bbox_std=stds,
                class_agnostic=class_agnostic,
                bbox_decode_type=bbox_decode_type
            )
            o2 = decode_bbox(
                mx.ndarray,
                anchors,
                agnostic_deltas if class_agnostic else deltas,
                im_infos,
                means,
                stds,
                class_agnostic,
                bbox_decode_type=bbox_decode_type
            )

            print(o1)
            print(o2.shape)
            print(o1 - o2)
