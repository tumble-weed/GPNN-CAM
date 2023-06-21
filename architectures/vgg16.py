#=================================================
# bypasses the flattening layer
#=================================================
# %debug
import torchvision
from torchvision.models.vgg import (VGG,cfgs,
                                    # load_state_dict_from_url,
                                make_layers)
from torchvision.models.vgg import model_urls as model_urls_vgg
from typing import Union, List, Dict, Any, cast
import torch

import os
import sys
import re
import warnings
from urllib.parse import urlparse  # noqa: F401
from torch.hub import download_url_to_file,_is_legacy_zip_format,_legacy_zip_load,get_dir
import errno
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location)
    return torch.load(cached_file, map_location=map_location)


class MyVGG(VGG):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # import pdb;pdb.set_trace()
        # why no avgpool: https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/torchray/benchmark/models.py#L103
        # x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = MyVGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls_vgg[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

