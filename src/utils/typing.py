from typing import Annotated

import torch
from typing_extensions import Generic, TypeVarTuple

Ts = TypeVarTuple("Ts")


class Tensor(Generic[*Ts], torch.Tensor):
    pass


BATCH = B = Annotated[int, "batch_size"]
SEQUENCE = T = Annotated[int, "length"]
HIDDEN_DIM = D = Annotated[int, "hidden_dim"]
LAYER = L = Annotated[int, "layer"]
DUMMY = X = Annotated[int, "dummy"]

HEAD = Annotated[int, "head"]
HEAD_KV = Annotated[int, "head_kv"]
HEAD_DIM = Annotated[int, "head_dim"]
HEAD_DIM_DIV_2 = Annotated[int, "head_dim_div_2"]
LAYER_PLUS_1 = Annotated[int, "layer_plus_1"]
