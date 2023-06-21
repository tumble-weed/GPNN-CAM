def voc_as_mask(label, class_id):
    # copied from https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/torchray/benchmark/datasets.py#L98
    """Convert a VOC detection label to a mask.

    Return a boolean mask selecting the region contained in the bounding boxes
    of :attr:`class_id`.

    Args:
        label (dict): an image label in the VOC detection format.
        class_id (int): ID of the requested class.

    Returns:
        :class:`torch.Tensor`: 2D boolean tensor.
    """
    weight, height = voc_as_image_size(label)
    mask = torch.zeros((height, weight), dtype=torch.uint8)
    objs = label['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        this_class_id = _VOC_CLASS_TO_INDEX[obj['name']]
        if this_class_id != class_id:
            continue
        bbox = obj['bndbox']
        ymin = int(bbox['ymin']) - 1
        ymax = int(bbox['ymax']) - 1
        xmin = int(bbox['xmin']) - 1
        xmax = int(bbox['xmax']) - 1
        mask[ymin:ymax + 1, xmin:xmax + 1] = 1
        
    if version.parse(torch.__version__) >= version.parse("1.2.0"):
        mask = mask.to(torch.bool)
    return mask
def evaluate(self, mask, point):
    # copied from https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/torchray/benchmark/pointing_game.py#LL58C11-L58C11
    r"""Evaluate a point prediction.

    The function tests whether the prediction :attr:`point` is within a
    certain tolerance of the object ground-truth region :attr:`mask`
    expressed as a boolean occupancy map.

    Use the :func:`reset` method to clear all counters.

    Args:
        mask (:class:`torch.Tensor`): :math:`\{0,1\}^{H\times W}`.
        point (tuple of ints): predicted point :math:`(u,v)`.

    Returns:
        int: +1 if the point hits the object; otherwise -1.
    """
    # Get an acceptance region around the point. There is a hit whenever
    # the acceptance region collides with the class mask.
    v, u = torch.meshgrid((
        (torch.arange(mask.shape[0],
                        dtype=torch.float32) - point[1])**2,
        (torch.arange(mask.shape[1],
                        dtype=torch.float32) - point[0])**2,
    ))
    accept = (v + u) < self.tolerance**2

    # Test for a hit with the corresponding class.
    hit = (mask & accept).view(-1).any()

    return +1 if hit else -1