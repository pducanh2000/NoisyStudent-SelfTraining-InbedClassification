import sys
from collections import defaultdict
import torch

_model_entrypoints = {}  # mapping of model name to entrypoint fns


def register_model(fn):
    _model_entrypoints[fn.__name__] = fn

    return fn


def create_model(model_name, pretrained=None, checkpoint_path=None, **kwargs):
    print(_model_entrypoints)
    if model_name in _model_entrypoints.keys():
        create_fn = _model_entrypoints[model_name]
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    model = create_fn(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    return model


if __name__ == '__main__':
    print(_model_entrypoints)
