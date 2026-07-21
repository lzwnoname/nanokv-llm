"""
AWQ 量化推理性能基准：测量 TTFT / TPOT / 端到端时延 / 吞吐 / 峰值显存。

设计原则（控制变量）：
- 一次进程只测一个配置（TP 会 init_process_group，不能在同进程内切换模型），
  baseline 与 AWQ 分开运行，对比两次结果得到加速比。
- 关闭 KV 压缩（kv_compress_enabled=False），排除压缩逻辑对 decode 计时的干扰。
- batch=1 单请求测 TTFT/TPOT：prefill step 耗时即 TTFT，
  每个 decode step 只产 1 个 token，decode step 平均耗时即 TPOT。
- ignore_eos=True + 固定 max_tokens，保证 decode 步数固定、可比。
- 前 warmup 次结果丢弃，取剩余的中位数。

用法示例：
  # baseline (FP16)
  python bench_awq.py --model ~/huggingface/Qwen3-8B --quant none \
      --input_len 512 --output_len 256

  # AWQ (torch kernel)
  python bench_awq.py --model ~/huggingface/Qwen3-8B-AWQ --quant awq --awq_kernel torch \
      --input_len 512 --output_len 256

  # AWQ (triton kernel)
  python bench_awq.py --model ~/huggingface/Qwen3-8B-AWQ --quant awq --awq_kernel triton \
      --input_len 512 --output_len 256
"""
import os
import time
import argparse
import statistics
from random import randint, seed

import torch

from nanokvllm import LLM, SamplingParams


def build_engine(args):
    kwargs = dict(
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        kv_compress_enabled=False,  # 测纯量化，关闭 KV 压缩
    )
    if args.quant == "awq":
        kwargs["quantization"] = "awq"
        kwargs["awq_kernel"] = args.awq_kernel
    return LLM(os.path.expanduser(args.model), **kwargs)


def measure_latency(engine, input_len, output_len, repeats=5, warmup=2):
    """batch=1 单请求，分离 TTFT / TPOT / E2E。"""
    seed(0)
    ttfts, tpots, e2es = [], [], []
    for r in range(repeats + warmup):
        prompt = [randint(0, 10000) for _ in range(input_len)]
        engine.add_request(
            prompt,
            SamplingParams(temperature=0.7, ignore_eos=True, max_tokens=output_len),
        )
        ttft = None
        decode_times = []
        torch.cuda.synchronize()
        e2e_start = time.perf_counter()
        while not engine.is_finished():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, num_tokens = engine.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if num_tokens > 0:          # prefill step
                ttft = dt
            else:                       # decode step（每步 1 token）
                decode_times.append(dt)
        e2e = time.perf_counter() - e2e_start
        if r < warmup:
            continue
        ttfts.append(ttft * 1000.0)                          # ms
        tpots.append(statistics.mean(decode_times) * 1000.0)  # ms/token
        e2es.append(e2e * 1000.0)                            # ms
    return {
        "TTFT_ms": statistics.median(ttfts),
        "TPOT_ms": statistics.median(tpots),
        "E2E_ms": statistics.median(e2es),
        "decode_tok_s": 1000.0 / statistics.median(tpots),
    }


def measure_throughput(engine, num_seqs, input_len, output_len):
    """大 batch 吞吐 + 峰值显存。"""
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    sps = [
        SamplingParams(temperature=0.7, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.perf_counter()
    engine.generate(prompts, sps, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    total_out = num_seqs * output_len
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    return {
        "gen_throughput_tok_s": total_out / dt,
        "time_s": dt,
        "peak_mem_GB": peak_mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", choices=["none", "awq"], default="none")
    parser.add_argument("--awq_kernel", choices=["torch", "triton"], default="torch")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--enforce_eager", action="store_true",
                        help="建议先开启以排除 CUDA graph 干扰，再关闭对比")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--input_len", type=int, default=512)
    parser.add_argument("--output_len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--throughput_bs", type=int, default=0,
                        help=">0 时额外跑一次该 batch 的吞吐测试")
    args = parser.parse_args()

    engine = build_engine(args)

    tag = f"quant={args.quant}"
    if args.quant == "awq":
        tag += f",kernel={args.awq_kernel}"
    tag += f",eager={args.enforce_eager},in={args.input_len},out={args.output_len}"

    print(f"\n===== Latency (batch=1) | {tag} =====")
    lat = measure_latency(engine, args.input_len, args.output_len,
                          repeats=args.repeats, warmup=args.warmup)
    for k, v in lat.items():
        print(f"  {k:>16}: {v:.3f}")

    if args.throughput_bs > 0:
        print(f"\n===== Throughput (batch={args.throughput_bs}) | {tag} =====")
        thr = measure_throughput(engine, args.throughput_bs,
                                 args.input_len, args.output_len)
        for k, v in thr.items():
            print(f"  {k:>20}: {v:.3f}")


if __name__ == "__main__":
    main()
