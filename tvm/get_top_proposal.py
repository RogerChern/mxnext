import mxnet as mx

def get_top_proposal(F=mx.ndarray, score=None, bbox=None, batch_size=None, top_n=None):
    with mx.name.Prefix("get_top_proposal: "):
        score = F.reshape(score, shape=[0, -3, -2])
        argsort_score = F.argsort(score, axis=-1, is_ascend=False)
        argsort_score = F.slice_axis(argsort_score, axis=-1, begin=0, end=top_n)
        arange_index = F.arange(0, batch_size).reshape([batch_size, 1])
        arange_index = F.broadcast_axis(arange_index, axis=1, size=top_n).reshape(-1)
        top_indexes = F.stack(arange_index, argsort_score.reshape(-1).astype("float32"))
        sort_score = F.gather_nd(score, top_indexes).reshape([-4, -1, top_n, 1])  # (#img, #proposal, 1)
        sort_bbox = F.gather_nd(bbox, top_indexes).reshape([-4, -1, top_n, 4])  # (#img, #proposal, 4)

    return sort_bbox, sort_score