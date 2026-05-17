import copy
import torch
import torch.utils.checkpoint as cp
from fvcore.nn import FlopCountAnalysis


def count_flops_and_params(
    model,
    img_size=200,      # H = W
    in_channels=5,
    device="cuda"
):
    """
    论文级 FLOPs / Params 统计
    - 关闭 checkpoint
    - dummy input 使用 zeros
    - deepcopy(model)，不污染训练
    """

    # ---------------------------------------------------
    # 1. deepcopy model，避免污染训练
    # ---------------------------------------------------
    model = copy.deepcopy(model).to(device)
    model.eval()

    # ---------------------------------------------------
    # 2. monkey-patch checkpoint（关键！）
    # ---------------------------------------------------
    _old_checkpoint = cp.checkpoint
    cp.checkpoint = lambda func, *args, **kwargs: func(*args)

    try:
        # ---------------------------------------------------
        # 3. dummy input（zeros，论文标准）
        # ---------------------------------------------------
        torch.manual_seed(0)
        dummy_x = torch.zeros(
            1, in_channels, img_size, img_size, device=device
        )

        # ---------------------------------------------------
        # 4. FLOPs
        # ---------------------------------------------------
        flops = FlopCountAnalysis(model, dummy_x)
        total_flops = flops.total()

        # ---------------------------------------------------
        # 5. Params（只统计可训练参数）
        # ---------------------------------------------------
        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    finally:
        # 恢复 checkpoint
        cp.checkpoint = _old_checkpoint

    return total_flops, total_params
