import math
import mxnet as mx


def _corner_to_center(F, boxes):
    boxes = F.slice_axis(boxes, axis=-1, begin=-4, end=None)
    xmin, ymin, xmax, ymax = F.split(boxes, axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) * 0.5
    y = ymin + (height - 1) * 0.5
    return x, y, width, height


def encode_rbbox(F, anchors, assigned_gt_bboxes, im_infos, means, stds):
    """
    encode rotated bounding boxes
    Args:
        F: mx.ndarray or mx.symbol
        anchors: (#img, #roi, #cls * 4)
        assigned_gt_bboxes: (#img, #roi, 5)
        im_infos: (#img, 3)
        means: (5, )
        stds: (5, )
    Returns:
        deltas: (#img, #roi, #cls * 5)
    """
    with mx.name.Prefix("encode_rbbox: "):
        x, y, w, h, r = F.split(assigned_gt_bboxes, axis=-1, num_outputs=5)
        ax, ay, aw, ah = _corner_to_center(F, anchors)  # anchor

        dx = (x - ax) / aw  # the r is in radian and the positive direction is clockwise
        dy = (y - ay) / ah
        dw = F.log(w / aw)
        dh = F.log(h / ah)
        dr = r % (2 * math.pi) / (2 * math.pi)

        dx = (dx - means[0]) / stds[0]
        dy = (dy - means[1]) / stds[1]
        dw = (dw - means[2]) / stds[2]
        dh = (dh - means[3]) / stds[3]
        dr = (dr - means[4]) / stds[4]

        deltas = F.concat(dx, dy, dw, dh, dr, dim=-1)
    return deltas


if __name__ == "__main__":
    anchors = mx.nd.array(
        [
            [10, 20, 55, 77],
            [10, 20, 50, 77],
        ]
    ).reshape([2, 1, 4])  # (#img, #roi, 4)
    im_infos = mx.nd.array(
        [
            [76, 72, 1],
            [76, 76, 1],
        ]
    ).reshape([2, 3])  # (#img, 3)
    deltas = mx.nd.array(
        [
            [
                [0.0, 0.0, 0.0, 0.1, 0.1],
                [0.5, 0.1, 0.2, 0.1, 0.3],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.1],
                [0.5, 0.1, 0.2, 0.1, 0.3],
            ],
        ]
    ).reshape([2, 1, 10])  # (#img, #roi, #cls * 5)
    gts = mx.nd.array(
        [
            [
                [45.84, 49.66, 48.35, 62.20, 1.06],
            ],
            [
                [41.89, 49.66, 43.10, 62.20, 1.06],
            ],
        ]
    ).reshape([2, 1, 5])  # (#img, #roi, 5)
    means = (0.04, 0.01, 0.03, 0.02, 0.02)
    stds = (0.5, 0.1, 0.1, 0.5, 0.5)

    o1 = encode_rbbox(
        mx.ndarray,
        anchors,
        gts,
        im_infos,
        means,
        stds,
    )

    print(o1)
    print(deltas)

