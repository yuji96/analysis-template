import contextlib
import re
from functools import partial

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from utils.typing import Tensor


class Hook:
    """Base class for hooks."""

    hook: RemovableHandle

    def __init__(
        self,
        module: nn.Module,
        to_cpu: bool = True,
        pre_hook: bool = False,
        pre_positional_args_keys: list[str] = None,
    ):
        self.args = None
        self.kwargs = None
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)
        self.to_cpu = to_cpu

    def hook_fn(self, module, args, kwargs, output) -> None:
        if self.to_cpu:
            args = tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        self.args = args
        self.kwargs = kwargs
        self.outputs = output

    @classmethod
    @contextlib.contextmanager
    def context(cls, hooks: "list[Hook]"):
        """Context manager to use the hook."""
        try:
            with torch.inference_mode():
                yield
        finally:
            for hook in hooks:
                hook.hook.remove()
