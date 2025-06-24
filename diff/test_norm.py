import paddle
import torch

from paddle.incubate.nn.functional import fused_rms_norm
from vllm.model_executor.layers.layernorm import rms_norm


def test(x, weight, eps=1e-5):
    weight_pd = paddle.to_tensor(weight.view(torch.int16).cpu().numpy(), dtype='int16').view(paddle.bfloat16).cuda()
    x_pd = paddle.to_tensor(x.view(torch.int16).cpu().numpy(), dtype='int16').view(paddle.bfloat16).cuda()

    print(f"x: {x.dtype}, {x.shape} {x.device}")
    print(f"weight: {weight.dtype}, {weight.shape} {weight.device}")
    # print(f"x_pd: {x_pd.dtype}, {x_pd.shape}")
    # print(f"weight_pd: {weight_pd.dtype}, {weight_pd.shape}")

    out_pd = fused_rms_norm(
        x_pd,
        norm_weight=weight_pd,
        norm_bias=None,
        epsilon=eps,
        begin_norm_axis=1,
        bias=None,
        residual=None,
        quant_scale=-1,
        quant_round_type=0,
        quant_max_bound=0,
        quant_min_bound=0,
    )[0]

    out_pt = rms_norm(x=x,
        weight=weight,
        variance_epsilon=eps,
    )

    out_pd = torch.tensor(out_pd.view(paddle.int16).cpu().numpy(), dtype=torch.int16).view(torch.bfloat16).cuda()
    print("out_cmp", (out_pd == out_pt).sum())

if __name__ == "__main__":
    h = 8192

    print("=" * 80, "\n", "test with ones.")
    x = torch.ones(1, h, dtype=torch.bfloat16).cuda()
    weight =  torch.ones(h, dtype=torch.bfloat16).cuda()
    test(x, weight)

    print("=" * 80, "\n", "test with aragne.")
    x = torch.arange(h, dtype=torch.bfloat16).cuda()[None,:]
    weight =  torch.arange(h, dtype=torch.bfloat16).cuda()
    test(x, weight)

    print("=" * 80, "\n", "test with rand(0,1).")
    x = torch.rand(h, dtype=torch.bfloat16).cuda()[None,:]
    weight =  torch.rand(h, dtype=torch.bfloat16).cuda()
    test(x, weight)

import pdb; pdb.set_trace()
