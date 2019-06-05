def _corner_to_center(F, boxes):
    boxes = F.slice_axis(boxes, axis=-1, begin=-4, end=None)
    xmin, ymin, xmax, ymax = F.split(boxes, axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) * 0.5
    y = ymin + (height - 1) * 0.5
    return x, y, width, height

def decode_bbox(F, anchors, deltas, im_infos, means, stds, class_agnostic):
    ctr_x, ctr_y, width, height = _corner_to_center(F, anchors)
    im_infos = F.split(im_infos, axis=-1, num_outputs=3, squeeze_axis=False)
    if class_agnostic:
        deltas = F.slice_axis(deltas, axis=-1, begin=4, end=None)
    if not class_agnostic:
        deltas = F.reshape(deltas, [0, 0, -4, -1, 4])
        ctr_x = F.expand_dims(ctr_x, axis=-1)
        ctr_y = F.expand_dims(ctr_y, axis=-1)
        width = F.expand_dims(width, axis=-1)
        height = F.expand_dims(height, axis=-1)
    dx, dy, dw, dh = F.split(deltas, axis=-1, num_outputs=4)

    dx = dx * stds[0] + means[0]
    dy = dy * stds[1] + means[1]
    dw = dw * stds[2] + means[2]
    dh = dh * stds[3] + means[3]

    pred_ctr_x = F.broadcast_add(F.broadcast_mul(dx, width), ctr_x)
    pred_ctr_y = F.broadcast_add(F.broadcast_mul(dy, height), ctr_y)
    pred_w = F.broadcast_mul(F.exp(dw), width)
    pred_h = F.broadcast_mul(F.exp(dh), height)

    x1 = pred_ctr_x - 0.5 * (pred_w - 1.0)
    y1 = pred_ctr_y - 0.5 * (pred_h - 1.0)
    x2 = pred_ctr_x + 0.5 * (pred_w - 1.0)
    y2 = pred_ctr_y + 0.5 * (pred_h - 1.0)

    x1 = F.maximum(F.broadcast_minimum(x1, im_infos[1] - 1.0), 0)
    y1 = F.maximum(F.broadcast_minimum(y1, im_infos[0] - 1.0), 0)
    x2 = F.maximum(F.broadcast_minimum(x2, im_infos[1] - 1.0), 0)
    y2 = F.maximum(F.broadcast_minimum(y2, im_infos[0] - 1.0), 0)

    out = F.concat(x1, y1, x2, y2, dim=-1)
    if not class_agnostic:
        out = F.reshape(out, [0, 0, -3, -2])
    return out
