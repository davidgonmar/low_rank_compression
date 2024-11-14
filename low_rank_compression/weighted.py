import torch
from low_rank_compression.modules import LowRankLinear
import torch.nn as nn
from typing import Callable
from tqdm import tqdm


def fischer_info(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    model.train()
    model.zero_grad()
    device = next(model.parameters()).device

    for batch in tqdm(dataloader, desc="Computing Fisher Information"):
        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

    with torch.no_grad():
        mean_grads = {
            name: param.grad.pow_(2).div_(len(dataloader))
            for name, param in model.named_parameters()
        }

    model.zero_grad()
    return mean_grads


# https://arxiv.org/abs/2207.00112
def linear_to_weighted_low_rank_linear(
    linear: nn.Linear, orig_grad: torch.Tensor, ratio: float
):
    # Summed by row
    summed_grad = orig_grad.sum(dim=0)
    assert summed_grad.shape == (linear.weight.shape[1],)
    II = torch.diag(summed_grad)
    II_inv = torch.inverse(II)
    # Original linear -> O = X @ W
    # Low rank linear -> W = U @ S @ V_T -> O = X @ U @ S @ V_T
    W, b = linear.weight.T, linear.bias
    W = II @ W
    U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD
    orig_rank = min(S.shape)
    rank = max(int(orig_rank * ratio), 16)
    S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
    # pad S to be {IN x OUT}
    in_f, out_f = W.shape
    assert S.shape == (rank, rank)
    assert U.shape == (in_f, in_f)
    assert V_T.shape == (out_f, out_f)
    W0 = II_inv @ U[:, :rank] @ S  # in R^{IN x RANK}
    W1 = V_T[:rank, :]  # in R^{RANK x OUT}
    low_rank_linear = LowRankLinear(
        linear.weight.shape[1],
        linear.weight.shape[0],
        rank,
        bias=linear.bias is not None,
    )
    low_rank_linear.w0.data = W0
    low_rank_linear.w1.data = W1
    if b is not None and linear.bias is not None:
        low_rank_linear.bias.data = b
    else:
        low_rank_linear.bias = None
    return low_rank_linear


def default_should_do(module: nn.Module, full_name: str):
    # embedding or lm head leave as is
    return isinstance(module, nn.Linear) and not any(
        x in full_name for x in ["embedding", "lm_head"]
    )


def _to_low_rank_recursive(model: nn.Module, should_do: Callable, prefix="", **kwargs):
    modules_to_replace = []

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            if should_do(module, full_name):
                modules_to_replace.append((full_name, module))
        else:
            modules_to_replace.extend(
                _to_low_rank_recursive(module, should_do, prefix=full_name, **kwargs)
            )

    return modules_to_replace


def set_nested_attr(obj, attr, value):
    attrs = attr.split(".")
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attrs[-1], value)


def to_low_rank(
    model: nn.Module, dataloader, should_do: Callable = default_should_do, **kwargs
):
    finfo = fischer_info(model, dataloader)
    modules_to_replace = _to_low_rank_recursive(
        model, should_do=should_do, prefix="", **kwargs
    )

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        low_rank_module = linear_to_weighted_low_rank_linear(
            module, finfo[name + ".weight"], ratio=0.3
        )
        set_nested_attr(model, name, low_rank_module)
        finfo.pop(name + ".weight")  # remove from dict to free up memory

    return model
