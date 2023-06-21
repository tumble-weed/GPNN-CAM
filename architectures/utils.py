# from torchray.benchmark.models import model_urls
import torch
from torch import nn
from collections import OrderedDict
from architectures.model_urls import model_urls

def replace_module(model, module_name, new_module):
    r"""Replace a :class:`torch.nn.Module` with another one in a model.
    Args:
        model (:class:`torch.nn.Module`): model in which to find and replace
            the module with the name :attr:`module_name` with
            :attr:`new_module`.
        module_name (str): path of module to replace in the model as a string,
            with ``'.'`` denoting membership in another module. For example,
            ``'features.11'`` in AlexNet (given by
            :func:`torchvision.models.alexnet.alexnet`) refers to the 11th
            module in the ``'features'`` module, that is, the
            :class:`torch.nn.ReLU` module after the last conv layer in
            AlexNet.
        new_module (:class:`torch.nn.Module`): replacement module.
    """
    return _replace_module(model, module_name.split('.'), new_module)


def _replace_module(curr_module, module_path, new_module):
    r"""Recursive helper function used in :func:`replace_module`.
    Args:
        curr_module (:class:`torch.nn.Module`): current module in which
            to search for the module with the relative path given by
            ``module_path``.
        module_path (list of str): path of module to replace in the model as
            a list, where each element is a member of the previous element's
            module. For example, ``'features.11'`` in AlexNet (given by
            :func:`torchvision.models.alexnet.alexnet`) refers to the 11th
            module in the ``'features'`` module, that is, the
            :class:`torch.nn.ReLU` module after the last conv layer in
            AlexNet.
        new_module (:class:`torch.nn.Module`): replacement module.
    """

    # TODO(ruthfong): Extend support to nn.ModuleList and nn.ModuleDict.
    if isinstance(curr_module, nn.Sequential):
        module_dict = OrderedDict(curr_module.named_children())
        assert module_path[0] in module_dict
        if len(module_path) == 1:
            submodule = new_module
        else:
            submodule = _replace_module(
                module_dict[module_path[0]],
                module_path[1:], new_module)
        if module_dict[module_path[0]] is not submodule:
            module_dict[module_path[0]] = submodule
            curr_module = nn.Sequential(module_dict)
    else:
        assert hasattr(curr_module, module_path[0])
        if len(module_path) == 1:
            submodule = new_module
        else:
            submodule = _replace_module(
                getattr(curr_module, module_path[0]),
                module_path[1:], new_module)
        setattr(curr_module, module_path[0], submodule)

    return curr_module