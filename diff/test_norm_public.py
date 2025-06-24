import paddle
import torch

from paddle.incubate.nn.functional import fused_rms_norm
from vllm.model_executor.layers.layernorm import rms_norm


def torch_fp32(x, weight, eps=1e-5):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def torch_bf16(x, weight, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x


def test(x, weight, eps=1e-5):
    weight_pd = paddle.to_tensor(weight.view(torch.int16).cpu().numpy(), dtype='int16').view(paddle.bfloat16).cuda()
    x_pd = paddle.to_tensor(x.view(torch.int16).cpu().numpy(), dtype='int16').view(paddle.bfloat16).cuda()

    out_paddle = fused_rms_norm(
        x_pd, norm_weight=weight_pd, norm_bias=None, epsilon=eps, begin_norm_axis=1, bias=None, 
        residual=None, quant_scale=-1, quant_round_type=0, quant_max_bound=0, quant_min_bound=0,
    )[0]
    out_vllm = rms_norm(x=x, weight=weight, variance_epsilon=eps,)
    out_paddle = torch.tensor(out_paddle.view(paddle.int16).cpu().numpy(), dtype=torch.int16).view(torch.bfloat16).cuda()

    out_torch_fp32 = torch_fp32(x=x, weight=weight)
    out_torch_bf16 = torch_bf16(x=x, weight=weight)

    print("[vLLM & Paddle]:      ", (out_vllm == out_paddle).sum().item())
    print("[vLLM & TorchFP32]:   ", (out_vllm == out_torch_fp32).sum().item())
    print("[vLLM & TorchBF16]:   ", (out_vllm == out_torch_bf16).sum().item())
    print("[Paddle & TorchFP32]: ", (out_paddle == out_torch_fp32).sum().item())
    print("[Paddle & TorchBF16]: ", (out_paddle == out_torch_bf16).sum().item())


if __name__ == "__main__":
    for h in [16, 64] * 10:
        print("=" * 80)
        print(f"[h={h}]")

        print("=> test with ones.")
        x = torch.ones(1, h, dtype=torch.bfloat16).cuda()
        weight =  torch.ones(h, dtype=torch.bfloat16).cuda()
        test(x, weight)

        print("test with aragne.")
        x = torch.arange(h, dtype=torch.bfloat16).cuda()[None,:]
        weight =  torch.arange(h, dtype=torch.bfloat16).cuda()
        test(x, weight)

        print("test with rand(0,1).")
        x = torch.rand(h, dtype=torch.bfloat16).cuda()[None,:]
        weight =  torch.rand(h, dtype=torch.bfloat16).cuda()
        test(x, weight)


