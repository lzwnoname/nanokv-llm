# Triton 入门与 AWQ Fused GEMM 算子教学文档

> **学习目标**：零基础掌握 Triton 编程模型的核心知识，并最终能独立写出 `nanokvllm/kernel/awq_triton_kernel.py` 里的 `awq_gemm_kernel`（int4 反量化 + GEMM 融合算子）。
> **学习方式**：6 个递进练习，从最简单的向量加法开始，每一步都在前一步基础上加一个新概念，最后拼装成目标算子。
> **前置要求**：会 PyTorch，了解矩阵乘法、GPU 有很多线程能并行计算即可，不需要任何 CUDA/GPU 编程经验。

---

## 0. 任务目标与验收标准

```
最终产物：nanokvllm/kernel/awq_triton_kernel.py 里的 awq_gemm_kernel

输入：
  x:       [M, K]         fp16     （M = batch*seq，decode 时 M 通常很小，甚至 M=1）
  qweight: [K, N//8]       int32    （int4 打包权重）
  qzeros:  [K//group, N//8] int32   （int4 打包 zero_point）
  scales:  [K//group, N]   fp16    （分组 scale）
输出：
  y:       [M, N]         fp16     （= dequant(qweight,qzeros,scales) 与 x 的矩阵乘）

验收标准：
  y 与 nanokvllm/layers/awq_gemm.py::AWQGemmTorch 算出的结果数值一致（atol=1e-2 量级，fp16 精度）
  且不能像 AWQGemmTorch 那样先反量化出完整 [K,N] fp16 权重再 matmul
  —— 必须在 kernel 内部边算边反量化（fused），这是它的存在意义
```

记住这个目标，后面每个练习都在为它做准备。

---

## 1. 为什么需要 Triton：一句话建立心智模型

CUDA 编程要手写线程、线程块、共享内存管理，非常繁琐。**Triton 是一个"用 Python 写 GPU kernel"的编译器**：你写一段看起来像 NumPy/PyTorch 的代码，Triton 编译器帮你自动做线程调度、内存合并访问、寄存器分配等底层优化。

**核心心智模型只有一句话**：

> Triton kernel 是一个**函数模板**，GPU 会同时启动很多个"实例"（program）并行运行这个函数，每个实例通过 `program_id` 知道"我是第几个"，从而计算出"我该处理数据的哪一部分"。

这和 PyTorch 的向量化思维完全不同——PyTorch 里你写 `x + y` 时不用关心"谁在哪个线程算哪个元素"，但写 Triton kernel 时，你要**亲自决定每个 program 该读哪块数据、写哪块数据**。这也是 Triton 比 PyTorch 算子快的原因：你可以针对具体硬件特性（内存合并访问、L2 缓存复用）手动优化数据搬运方式。

---

## 2. 练习 1：向量加法（Triton Hello World）

### 2.1 目标

实现 `z = x + y`，`x, y, z` 都是长度 `n` 的一维向量。

### 2.2 需要的新概念

| 概念 | 含义 |
|---|---|
| `@triton.jit` | 装饰器，标记这个函数会被编译成 GPU kernel |
| `tl.program_id(axis)` | 当前这个 program 是第几个（类似 CUDA 的 `blockIdx`） |
| `BLOCK_SIZE: tl.constexpr` | 每个 program 处理多少个元素，`constexpr` 表示编译期常量（会被编译器特化优化） |
| `tl.arange(0, BLOCK_SIZE)` | 生成 `[0, 1, ..., BLOCK_SIZE-1]` 的向量，类似 `np.arange` |
| `tl.load(ptr, mask=...)` | 从显存指针位置读数据到寄存器/共享内存，`mask` 决定哪些位置真的去读（越界的不读） |
| `tl.store(ptr, value, mask=...)` | 把计算结果写回显存 |
| `kernel[grid](...)` | 启动 kernel，`grid` 决定总共启动多少个 program |

### 2.3 完整代码

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 第几个 program？
    pid = tl.program_id(axis=0)

    # 这个 program 负责的元素范围是 [pid*BLOCK_SIZE, (pid+1)*BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # [BLOCK_SIZE] 个具体下标

    # 越界保护：最后一个 block 可能超出 n_elements
    mask = offsets < n_elements

    # 读数据（越界位置不读，返回 0，但反正后面也不会 store 出去）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    z = x + y

    # 写回显存
    tl.store(z_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    z = torch.empty_like(x)
    n_elements = z.numel()
    BLOCK_SIZE = 1024

    # grid：总共要启动多少个 program 才能覆盖所有元素？
    # ceil(n_elements / BLOCK_SIZE) 个
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z


# 验证
x = torch.randn(10000, device="cuda")
y = torch.randn(10000, device="cuda")
z_triton = add(x, y)
z_torch = x + y
assert torch.allclose(z_triton, z_torch)
print("练习1通过！")
```

### 2.4 关键点拆解

**`grid` 和 `pid` 的关系**：假设 `n_elements=10000`，`BLOCK_SIZE=1024`，`grid = ceil(10000/1024) = 10`。Triton 会启动 10 个 program，`pid` 依次是 `0, 1, 2, ..., 9`。每个 program 算 `offsets = pid*1024 + [0..1023]`。

```
pid=0: 处理下标 [0, 1024)
pid=1: 处理下标 [1024, 2048)
...
pid=9: 处理下标 [9216, 10000)   ← 最后一个 block 只有 784 个有效元素，剩下 240 个被 mask 掉
```

**为什么 `x_ptr` 是一个"指针"而不是 tensor**：Triton kernel 操作的是**显存地址**，你传进去的 PyTorch tensor 在 kernel 内部会被自动转成它的 GPU 显存首地址（一个指针）。`x_ptr + offsets` 就是"首地址 + 偏移量"，指向具体要读的显存位置。这跟你在 C 语言里做指针运算是一回事。

**练习**：把 `BLOCK_SIZE` 改成 256、512、2048，重新跑一遍，确认结果依然正确（这证明 `mask` 起了作用，不管 block 怎么切都不会越界）。

---

## 3. 练习 2：理解 2D 数据布局（为矩阵乘法铺路）

向量加法是 1D 的，但矩阵是 2D 的。这一节专门练习"怎么用 1D 的 `program_id` 定位 2D 数据"。

### 3.1 目标

实现一个 kernel：把一个 `[M, N]` 的矩阵**每一行都加上一个标量**（模拟"处理 2D tile"的思维）。

### 3.2 新概念

| 概念 | 含义 |
|---|---|
| `tl.program_id(axis=0)` / `axis=1` | Triton 支持最多 3 维的 grid，`axis` 指定看哪一维 |
| stride（步长） | 矩阵在显存里是"拉平"存的，`stride` 告诉你"换一行要跳过多少个元素" |
| 2D `offsets` | 用 `[:, None]` 和 `[None, :]` 的广播技巧构造二维下标网格 |

### 3.3 代码

```python
@triton.jit
def add_scalar_2d_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_m,             # 矩阵每行之间的显存步长（通常就是 N，如果矩阵是连续存储的）
    scalar,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)   # 当前 program 负责第几个"行 block"
    pid_n = tl.program_id(1)   # 当前 program 负责第几个"列 block"

    # 构造这个 program 负责的行下标、列下标
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # 用广播构造二维网格的显存偏移
    # row_offsets[:, None] : [BLOCK_M, 1]
    # col_offsets[None, :] : [1, BLOCK_N]
    # 相乘/相加后广播成 [BLOCK_M, BLOCK_N]
    offsets = row_offsets[:, None] * stride_m + col_offsets[None, :]

    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x + scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def add_scalar_2d(x: torch.Tensor, scalar: float):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    add_scalar_2d_kernel[grid](
        x, out, M, N, x.stride(0), scalar,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


x = torch.randn(100, 200, device="cuda")
out = add_scalar_2d(x, 3.0)
assert torch.allclose(out, x + 3.0)
print("练习2通过！")
```

### 3.4 关键点拆解

**为什么 grid 现在是二维 `(cdiv(M,BLOCK_M), cdiv(N,BLOCK_N))`**：整个矩阵被切成了很多个 `[BLOCK_M, BLOCK_N]` 的小方块（tile），每个 program 负责一个 tile。`pid_m` 是"第几行 tile"，`pid_n` 是"第几列 tile"。

**广播下标是本节最重要的技巧**，务必手推一遍：

```
假设 BLOCK_M=2, BLOCK_N=3
row_offsets = [5, 6]              形状 [2]
col_offsets = [10, 11, 12]        形状 [3]

row_offsets[:, None] = [[5],
                         [6]]      形状 [2, 1]

col_offsets[None, :] = [[10, 11, 12]]   形状 [1, 3]

row_offsets[:,None] * stride_m + col_offsets[None,:]  广播相加 → [2, 3]
= 每个 (i,j) 位置的值是 row_offsets[i]*stride_m + col_offsets[j]
= 这正是矩阵里 (row_offsets[i], col_offsets[j]) 这个元素的一维显存偏移
```

这一步是整个 Triton 矩阵编程的核心技巧，后面写 GEMM、写 int4 unpack 全部靠这个套路。**务必自己在纸上画一遍这个广播过程**，直到完全理解为止。

---

## 4. 练习 3：矩阵乘法（GEMM）—— tile 化思想

这是最关键的一节，理解了它，AWQ kernel 就只剩下"加个反量化步骤"这一点新知识了。

### 4.1 目标

实现 `C[M,N] = A[M,K] @ B[K,N]`，用标准的 tile-based 分块矩阵乘法。

### 4.2 新概念

| 概念 | 含义 |
|---|---|
| `tl.dot(a, b)` | Triton 内建的矩阵乘法原语，`a: [BLOCK_M, BLOCK_K]`，`b: [BLOCK_K, BLOCK_N]`，输出 `[BLOCK_M, BLOCK_N]`，底层用 Tensor Core 加速 |
| accumulator（累加器） | GEMM 要沿 K 维累加很多次，需要一个变量不断累加中间结果 |
| K 方向循环 | 一个 tile 装不下整个 K 维，要分批读、分批累加 |
| `tl.zeros(shape, dtype)` | 创建初始为 0 的累加器 |

### 4.3 GEMM 的分块思想图解

```
C[M,N] = A[M,K] @ B[K,N]

把 C 切成很多个 [BLOCK_M, BLOCK_N] 的小方块，每个 program 负责算一个方块。

要算 C 的这个方块，需要：
  A 的对应行条带 [BLOCK_M, K]（M 方向对齐，K 方向全部）
  B 的对应列条带 [K, BLOCK_N]（N 方向对齐，K 方向全部）

但 K 可能很大（比如 4096），一次性读进 SRAM 装不下，
所以要把 K 维也切成 BLOCK_K 一段一段地读：

for k_start in range(0, K, BLOCK_K):
    a_tile = A[BLOCK_M行, k_start:k_start+BLOCK_K]     # [BLOCK_M, BLOCK_K]
    b_tile = B[k_start:k_start+BLOCK_K, BLOCK_N列]     # [BLOCK_K, BLOCK_N]
    accumulator += a_tile @ b_tile                      # tl.dot

最后把 accumulator 写回 C 的对应方块
```

### 4.4 代码

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,      # A 的行步长、列步长
    stride_bk, stride_bn,      # B 的行步长、列步长
    stride_cm, stride_cn,      # C 的行步长、列步长
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 本 program 负责的 C 方块的行、列下标
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)                      # [BLOCK_K]，K 方向的局部下标，每轮循环复用

    # A、B 的初始 tile 指针（后面循环里不断平移）
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        b_mask = (rk[:, None] + k < K) & (rn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

        # 把 A、B 的指针沿 K 方向平移一个 BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
b = torch.randn(256, 512, device="cuda", dtype=torch.float16)
c_triton = matmul(a, b)
c_torch = a @ b
assert torch.allclose(c_triton, c_torch, atol=1e-1, rtol=1e-2)   # fp16 精度较低，容差要放宽
print("练习3通过！")
```

### 4.5 关键点拆解

**为什么 `accumulator` 用 `float32` 而不是 `float16`**：矩阵乘法里累加很多项，fp16 精度低容易累积误差，业界惯例是"输入 fp16，累加用 fp32，最后转回 fp16 输出"。这一点在写 AWQ kernel 时同样适用。

**指针平移 `a_ptrs += BLOCK_K * stride_ak`**：这是一个重要技巧——不用每次循环重新算 `rk + k` 再乘 stride，而是直接把上一轮的指针沿 K 方向"挪一格"，减少重复计算。

**`tl.dot` 对形状有要求**：通常要求 `BLOCK_M/BLOCK_N/BLOCK_K` 是 16 的倍数（Tensor Core 硬件对齐要求），后面设计 AWQ kernel 的 BLOCK 大小时要记住这一点。

**练习**：把 `BLOCK_K` 改成 16、64、128 分别测一下正确性（性能先不管，练习阶段只关注正确性）。

---

## 5. 练习 4：位运算解包 int4（Triton 里怎么做 bit trick）

现在补最后一块拼图：**怎么在 Triton kernel 内部把 int32 解包成 8 个 int4**。这一步在 PyTorch 层面你已经写过（`awq_gemm.py::dequantize_awq_weight`），现在要搬到 Triton 里，用的是 GPU 原生支持的位运算指令，速度更快。

### 5.1 新概念

| 概念 | 含义 |
|---|---|
| `tl.load` 出来的 `int32` 值 | 可以直接用 Python 位运算符 `&`、`>>` 操作（Triton 支持标准位运算） |
| bitwise AND `& 0xF` | 取最低 4 位（一个 nibble） |
| 右移 `>> 4` | 把高 4 位移到低位，再 `& 0xF` 取出 |
| 二维广播位运算 | 用 `packed[:, None] >> shifts[None, :]` 一次性算出全部 8 个 nibble，避免 for 循环里多次 store |
| int4 → fp16 反量化公式 | `(q - zero) * scale`，和你在 `dequantize_awq_weight` 里写的一样 |

### 5.2 练习：写一个"解包单个 int32 里的 8 个 int4"的小 kernel

**反面教材（先看低效写法，理解为什么不能这样写）**：

```python
# ❌ 低效写法：for 循环里 8 次独立 store
for i in range(8):
    val = (packed >> (i * 4)) & 0xF
    out_offsets = offsets * 8 + i
    tl.store(out_ptr + out_offsets, val, mask=mask)
```

虽然 `range(8)` 是 constexpr，Triton 编译器会展开这个循环，但展开后是 **8 条独立的 `tl.store`**，不保证能合并成一条向量化写回指令。这在数据量小时不明显，但在正式 AWQ kernel 的 K 维循环里每次都这样写，累计的 store 开销会显著拖慢性能。

**推荐写法：二维广播 + 一次 store**

```python
@triton.jit
def unpack_int4_kernel(
    packed_ptr,     # int32, [N]
    out_ptr,        # int32, [N, 8]  存放解包后的 8 个 int4 值
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # [BLOCK_SIZE]
    mask = offsets < N

    packed = tl.load(packed_ptr + offsets, mask=mask)       # [BLOCK_SIZE] int32

    # 一次性算出全部 8 个 nibble（AutoAWQ 顺序拼接：低位在前）
    shifts = tl.arange(0, 8) * 4                             # [8] = [0, 4, 8, ..., 28]
    # packed[:, None] : [BLOCK_SIZE, 1]
    # shifts[None, :] : [1, 8]
    # 广播后一次得到 [BLOCK_SIZE, 8] 的全部 nibble
    all_nibbles = (packed[:, None] >> shifts[None, :]) & 0xF  # [BLOCK_SIZE, 8]

    # 一次性 store（用二维广播下标，跟练习 2 学的套路一样）
    out_offsets = offsets[:, None] * 8 + tl.arange(0, 8)[None, :]  # [BLOCK_SIZE, 8]
    tl.store(out_ptr + out_offsets, all_nibbles, mask=mask[:, None])


def unpack_int4(packed: torch.Tensor):
    N = packed.numel()
    out = torch.empty((N, 8), dtype=torch.int32, device=packed.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    unpack_int4_kernel[grid](packed, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(-1)


# 验证：用一个已知的 int32 反推它应该解包出的 8 个值
packed = torch.tensor([0x76543210], dtype=torch.int32, device="cuda")   # 手动构造
# 0x76543210 的 8 个 nibble（从低位到高位）应该是 0,1,2,3,4,5,6,7
result = unpack_int4(packed)
expected = torch.tensor([0,1,2,3,4,5,6,7], dtype=torch.int32, device="cuda")
assert torch.equal(result, expected)
print("练习4通过！")
```

### 5.3 关键点拆解

**为什么 `0x76543210` 解包出来是 `[0,1,2,3,4,5,6,7]`**：十六进制的每一位对应 4 个 bit（正好是一个 int4）。`0x76543210` 从**低位**（最右边）读起：`0, 1, 2, 3, 4, 5, 6, 7`。`packed >> (i*4) & 0xF` 就是取第 `i` 个（从低位数）nibble。

**二维广播位运算是本节的核心技巧**：回顾练习 2 里学的 `row_offsets[:, None] * stride + col_offsets[None, :]` 套路，这里完全一样——把 `packed`（一维）和 `shifts`（一维）通过 `[:, None]` / `[None, :]` 广播成二维，一次位运算就得到 `[BLOCK_SIZE, 8]` 的全部结果。这比 for 循环里 8 次独立 `tl.store` 更高效，因为：

1. **计算向量化**：一次广播位运算，GPU 可以用 SIMD 指令并行处理所有 8 个 nibble
2. **store 合并**：一条 `tl.store` 写回 `[BLOCK_SIZE, 8]`，编译器更容易生成向量化/合并的写回指令
3. **可预测**：不依赖编译器是否能合并循环展开后的 store，性能行为更确定

**这个技巧在正式 AWQ kernel 里的意义**：AWQ fused kernel 的 K 维循环每轮都要解包一个 `[BLOCK_K, BLOCK_N//8]` 的 qweight tile，如果用 for 循环写 8 次 store，每轮 K 循环都多 8 条 store 指令，累计开销显著。用二维广播一次算一次写，是这个 kernel 能做到高性能的关键细节之一。

**这跟你 PyTorch 版本的顺序要对齐**：回顾你在 `dequantize_awq_weight` 里写的 `torch.cat([low_value, high_value], dim=-1)`——这是"低4位、高4位"两两一组按 byte 处理的顺序。你需要用这个小练习**反过来验证**：用一个已知 qweight 小样本，同时跑 PyTorch 版本和这个 Triton 版本，确认两者解包出的 int4 序列完全一致。这是接下来写正式 kernel 前**必须做的正确性校验**，否则最终反量化结果会全错但很难发现。

**练习（重要）**：请你自己写一段代码，构造一个已知的 `qweight` 张量（可以是 `torch.arange` 生成后 view 成 int32），分别用 PyTorch 版 `dequantize_awq_weight` 和这个 Triton 小 kernel 解包，**打印比较两者展开出来的 int4 序列顺序**，确保完全对齐。如果不对齐，说明你的 PyTorch 版本或者这个 Triton 版本的位运算顺序需要调整（比如高低位换一下、pack_factor 内部顺序换一下）。这是本练习里最重要的一步，不要跳过。

---

## 6. 练习 5：拼装 —— Fused Dequant + GEMM（正式产物的骨架）

现在把练习 3（GEMM tile 化）和练习 4（int4 解包）拼在一起。这一节**只给骨架和关键步骤提示，具体代码由你自己填写**——这是检验你是否真正掌握前面知识的关键一步。

### 6.1 目标回顾

```
x:       [M, K]           fp16
qweight: [K, N//pack_factor]   int32   (pack_factor=8, 4bit)
qzeros:  [K//group_size, N//pack_factor]  int32
scales:  [K//group_size, N]     fp16

输出 y = x @ dequant(qweight, qzeros, scales)   形状 [M, N]
```

### 6.2 与练习 3 的核心区别

| 练习 3（纯 fp16 GEMM） | 目标算子（fused AWQ GEMM） |
|---|---|
| `b_ptrs` 直接指向 fp16 的 B 矩阵 | `qweight_ptrs` 指向 int32 打包的权重，形状是 `[K, N//8]` 不是 `[K, N]` |
| `b = tl.load(b_ptrs)` 直接得到能用的数值 | load 出来的是 int32，需要**在 kernel 内部当场解包成 int4，再反量化成 fp16**，才能喂给 `tl.dot` |
| 不需要 scale/zero | 每次读一个 K-tile 的权重，都要同步算出对应的 scale/zero（按 `group_size` 分组） |

### 6.3 分步骤设计提示（请照着这个顺序，自己写代码）

**Step 1：确定 BLOCK_K 和 group_size 的关系**

为了简化设计，**建议让 `BLOCK_K` 是 `group_size` 的整数倍**（比如 `group_size=128`，`BLOCK_K=128` 或 `64`）。这样每个 K-tile 内部的 scale/zero 是固定的（同属一个或几个 group），不会出现"一个 tile 内部跨越多个 group、需要分段处理"的复杂情况。先用最简单的设定：`BLOCK_K = group_size`。

**Step 2：想清楚 `qweight` 这个 tile 该怎么定位**

练习 3 里 B 矩阵 tile 定位公式是：
```python
b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
```
现在 `qweight` 的形状是 `[K, N//pack_factor]`，你要读的仍然是 `[BLOCK_K, BLOCK_N]` 大小的**逻辑权重**tile，但物理上只需要读 `[BLOCK_K, BLOCK_N//pack_factor]` 个 int32（因为每个 int32 装了 8 个值）。

**提示**：`BLOCK_N` 必须是 `pack_factor`（8）的整数倍，这样"逻辑 tile"和"物理 int32 tile"能对齐切分。

```python
# 物理 tile 的列下标（除以 pack_factor 后的坐标系）
rn_packed = pid_n * (BLOCK_N // pack_factor) + tl.arange(0, BLOCK_N // pack_factor)
qweight_ptrs = qweight_ptr + rk[:, None] * stride_qw_k + rn_packed[None, :] * stride_qw_n
```

**Step 3：load 出 int32，解包成 int4**

```python
qw_packed = tl.load(qweight_ptrs, mask=...)   # [BLOCK_K, BLOCK_N//8]  int32

# 解包：对每个 int32 取出 8 个 int4，拼成 [BLOCK_K, BLOCK_N]
# 提示：用练习4学到的位运算，配合 tl.arange(0,8) 广播出 8 个 shift 量，
#      再想办法把 [BLOCK_K, BLOCK_N//8] 和 "8个nibble" 这两个维度合并/交织成 [BLOCK_K, BLOCK_N]
# 这一步是本练习最大的难点，需要你综合运用练习2的广播技巧 + 练习4的位运算
```

*给一个思路方向（不是完整答案）*：可以先解包出形状 `[BLOCK_K, BLOCK_N//8, 8]`（用一个额外维度装 8 个 nibble），再 reshape/permute 成 `[BLOCK_K, BLOCK_N]`。Triton 的 reshape 能力有限，可能需要用 `tl.interleave` 或者手动构造下标的方式实现"交织"，这是需要你自己查 Triton 文档（`tl.interleave`、`tl.reshape`）去解决的部分。

**Step 4：定位并 load 对应的 scale / zero**

因为 `BLOCK_K = group_size`，这个 K-tile 正好对应**一个** group，所以 scale/zero 的定位很简单：

```python
group_id = k // group_size            # 当前是第几个 group（k 是本轮循环的 K 起始位置）
scale_ptrs = scales_ptr + group_id * stride_scale_g + rn[None, :] * stride_scale_n
scales_tile = tl.load(scale_ptrs, mask=...)     # [1, BLOCK_N] 广播到 [BLOCK_K, BLOCK_N]

# zeros 同理，但 zeros 也是 packed 的，需要先解包（同 Step 3 的方法，但只有 1 行需要解包）
```

**Step 5：反量化 + 累加进 GEMM**

```python
w_fp16 = (int4_vals.to(tl.float32) - zeros_vals.to(tl.float32)) * scales_tile.to(tl.float32)
w_fp16 = w_fp16.to(tl.float16)

x_tile = tl.load(x_ptrs, mask=...)              # [BLOCK_M, BLOCK_K]，跟练习3的 A 矩阵一样普通
accumulator += tl.dot(x_tile, w_fp16)
```

**Step 6：循环 + 收尾**

跟练习 3 完全一样——`for k in range(0, K, BLOCK_K)`，每轮把指针平移，循环完把 `accumulator` 转 fp16 写回输出。

### 6.4 完整骨架（含 TODO，请自己补全 TODO 部分）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def awq_gemm_kernel(
    x_ptr, qweight_ptr, qzeros_ptr, scales_ptr, y_ptr,
    M, N, K,
    group_size, pack_factor: tl.constexpr,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,     # 建议设置为等于 group_size
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # ---- 1. 读 x tile（跟练习3的 A 矩阵一样） ----
        x_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # ---- 2. 读 qweight tile 并解包成 int4 ----
        # TODO: 参照 6.3 Step 2-3，构造 qweight_ptrs，load，解包成 [BLOCK_K, BLOCK_N]

        # ---- 3. 读 scale / zero（当前 K-tile 对应第几个 group？） ----
        # TODO: 参照 6.3 Step 4

        # ---- 4. 反量化成 w_fp16 ----
        # TODO: 参照 6.3 Step 5

        # ---- 5. 累加 ----
        accumulator += tl.dot(x_tile, w_fp16)

        # ---- 6. 指针平移 ----
        x_ptrs += BLOCK_K * stride_xk
        # TODO: qweight_ptrs 也要平移

    y = accumulator.to(tl.float16)
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def awq_gemm_kernel_launch(x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    M, K = x.shape
    K2, N_packed = qweight.shape
    N = N_packed * pack_factor
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, group_size   # 先用小 tile 保证正确性，后面再调优
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    awq_gemm_kernel[grid](
        x, qweight, qzeros, scales, y,
        M, N, K,
        group_size, pack_factor,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    if bias is not None:
        y += bias
    return y
```

### 6.5 验收测试（你写完 TODO 之后，用这个测试验证）

```python
from nanokvllm.layers.awq_gemm import dequantize_awq_weight

def test_awq_gemm_kernel():
    torch.manual_seed(0)
    M, K, N = 8, 256, 512
    group_size, bits = 128, 4
    pack_factor = 32 // bits

    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    qweight = torch.randint(0, 2**31 - 1, (K, N // pack_factor), device="cuda", dtype=torch.int32)
    qzeros = torch.randint(0, 2**31 - 1, (K // group_size, N // pack_factor), device="cuda", dtype=torch.int32)
    scales = torch.rand(K // group_size, N, device="cuda", dtype=torch.float16) * 0.1

    # 参考答案：先反量化再 matmul
    w_ref = dequantize_awq_weight(qweight, qzeros, scales, group_size, pack_factor)
    y_ref = x @ w_ref

    # 你写的 fused kernel
    y_triton = awq_gemm_kernel_launch(x, qweight, qzeros, scales, group_size, pack_factor)

    assert torch.allclose(y_triton, y_ref, atol=1e-1, rtol=1e-2), \
        f"最大误差: {(y_triton - y_ref).abs().max().item()}"
    print("AWQ fused kernel 验证通过！")

test_awq_gemm_kernel()
```

---

## 7. 练习 6（选修）：性能调优基础知识

功能写对之后，如果想进一步学习怎么把它调快，可以了解这些概念（暂不要求现在就实现，先把 Phase 3 功能跑通即可）：

| 概念 | 作用 |
|---|---|
| `@triton.autotune` | 自动尝试多组 `BLOCK_M/BLOCK_N/BLOCK_K/num_warps` 组合，选运行最快的配置，避免手动调参 |
| `num_warps` | 每个 program 用多少个 warp（32 线程一组）并行执行，影响并行度和资源占用的平衡 |
| L2 cache 复用（`GROUP_SIZE_M` 分组调度） | 调整 program 的执行顺序，让相邻 program 复用同一块显存数据，减少重复搬运（这是官方 matmul 教程里的经典优化技巧） |
| `tl.multiple_of` / `tl.max_contiguous` | 给编译器提示"这个指针地址一定是 XX 的倍数/连续的"，帮助生成更快的访存指令 |

`@triton.autotune` 的使用范式（了解即可，不急着用）：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8),
        # ... 更多候选配置
    ],
    key=["M", "N", "K"],   # 当 M/N/K 变化时重新选择最优配置
)
@triton.jit
def awq_gemm_kernel(...):
    ...
```

### 7.1 L2 Cache 复用：`GROUP_SIZE_M` 分组调度详解

这一节展开讲练习 6 表格里提到的 `GROUP_SIZE_M` 技巧——**这是 Triton 官方 matmul 教程里最经典、性价比最高的优化**，只需要改 `pid_m`/`pid_n` 的计算方式，不用改 kernel 的计算逻辑本身，属于"低成本、高收益"的调优手段，值得单独学一遍。

#### 7.1.1 先搞清楚问题出在哪：GPU 是怎么调度 program 的

回顾练习 3 里 `grid = (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N))`。你写的 `grid` 是一个二维网格，但 GPU 实际执行时，会把这个二维网格**按行优先（row-major）拉平成一维**，依次编号 `0, 1, 2, 3, ...` 去调度：

```
假设 grid = (4, 6)，即 4 行 × 6 列 = 24 个 program

默认（行优先）调度顺序：
  pid=0  → (pid_m=0, pid_n=0)
  pid=1  → (pid_m=0, pid_n=1)
  pid=2  → (pid_m=0, pid_n=2)
  pid=3  → (pid_m=0, pid_n=3)
  pid=4  → (pid_m=0, pid_n=4)
  pid=5  → (pid_m=0, pid_n=5)
  pid=6  → (pid_m=1, pid_n=0)     ← 走完第 0 行才换到第 1 行
  ...
```

也就是说，**默认调度会先把 `pid_m=0` 这一整行的 program 全部跑完，才会跑到 `pid_m=1`**。

#### 7.1.2 这个默认顺序为什么浪费 L2 Cache

回顾练习 3，每个 program 要读：
- A 矩阵的一个行条带 `A[BLOCK_M行, :]`（只依赖 `pid_m`）
- B 矩阵的一个列条带 `B[:, BLOCK_N列]`（只依赖 `pid_n`）

GPU 有一块**所有 SM（流处理器）共享**的 L2 Cache。如果两个"背靠背"执行的 program 用到了同一块 A 或 B 的数据，第二个 program 就能直接从 L2 Cache 里读，不用再从显存（HBM，比 L2 慢得多）搬一次。

问题来了：默认顺序下，`pid=0` 到 `pid=5`（同一行 `pid_m=0`）虽然共用同一块 `A[0行, :]`，能复用 A；但它们分别要读 **6 块不同的 B**（`pid_n=0..5`）。等到 `pid=6` 换到 `pid_m=1` 时，`A[0行,:]` 早就被挤出 L2 Cache 了（因为中间读了 6 块不同的 B，L2 Cache 容量有限）。

**核心问题**：默认顺序下，同一批"背靠背执行"的 program，只能复用 A 或只能复用 B（不可能同时复用两者），因为 `pid_m` 要连续走完一整行才换，跨度太大。

#### 7.1.3 GROUP_SIZE_M 的核心思想：把 "一行" 拆成 "小方块"

`GROUP_SIZE_M` 的做法是：**不再按"一整行"分组，而是把 M 方向切成若干个 `GROUP_SIZE_M` 行一组的小分组，在这个小分组内，让 program 按"列优先"依次遍历完这个分组内所有的 `(pid_m, pid_n)` 组合，再换下一个分组**。

```
假设 GROUP_SIZE_M = 2（把 M 方向每 2 行分成一组）

grid 还是 (4, 6)，但调度顺序变成：

第 0 组（pid_m ∈ {0,1}）：
  pid=0  → (pid_m=0, pid_n=0)
  pid=1  → (pid_m=1, pid_n=0)     ← 注意！先把 pid_m=0,1 都跑完 pid_n=0，再换 pid_n
  pid=2  → (pid_m=0, pid_n=1)
  pid=3  → (pid_m=1, pid_n=1)
  pid=4  → (pid_m=0, pid_n=2)
  pid=5  → (pid_m=1, pid_n=2)
  ...（pid_n 继续跑到 5，共 2×6=12 个 program）

第 1 组（pid_m ∈ {2,3}）：
  pid=12 → (pid_m=2, pid_n=0)
  pid=13 → (pid_m=3, pid_n=0)
  ...
```

**为什么这样更省显存带宽**：看第 0 组里 `pid=0` 和 `pid=1` 这两个**背靠背**执行的 program——它们的 `pid_n` 都是 0（复用同一块 B 列条带），`pid_m` 不同（各自读不同的 A 行条带）。同时看 `pid=0` 和 `pid=2`（跨了 2 步）——`pid_m` 都是 0（复用同一块 A 行条带）。

也就是说，`GROUP_SIZE_M` 把"矩形网格"重新排列成"一组一组的小方阵"来遍历，让相邻执行的 program **同时**在 A 方向和 B 方向都有更高概率复用 L2 Cache 里的数据，而不是像默认顺序那样只能复用一边。`GROUP_SIZE_M` 越大，一个分组里可复用的 A 行条带越多，但分组太大也会让 L2 Cache 装不下这么多不同的 A/B 条带，所以这是一个需要调参的超参数（官方教程常用 `GROUP_SIZE_M=8`）。

#### 7.1.4 代码实现：怎么把 `pid` 重新映射成分组顺序的 `(pid_m, pid_n)`

这是本节唯一需要记住的代码套路，直接加在 kernel 开头、替换掉原来"直接用 `program_id` 当 `pid_m`/`pid_n`"的写法：

```python
@triton.jit
def awq_gemm_kernel(
    ...,
    GROUP_SIZE_M: tl.constexpr,   # 新增的调优参数，典型值 8
):
    pid = tl.program_id(axis=0)   # 注意：grid 现在改成一维！见下方启动代码的变化

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # 每个 group 里有多少个 program？(GROUP_SIZE_M 行) x (所有列)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # 当前 pid 属于第几个 group
    group_id = pid // num_pid_in_group

    # 这个 group 从第几行开始（最后一个 group 可能不满 GROUP_SIZE_M 行，要 min 一下）
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # 在 group 内部，重新算出真正的 pid_m 和 pid_n
    # pid 在 group 内部的相对位置
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # ---- 后面的代码跟练习 3 完全一样，只是 pid_m / pid_n 换成了上面重新算出来的值 ----
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ...
```

**launch 代码的变化**：因为现在 kernel 内部自己用一维 `pid` 反推 `pid_m`/`pid_n`，所以外部启动时 `grid` 也要改成一维（乘法算出总 program 数）：

```python
def awq_gemm_kernel_launch(x, qweight, qzeros, scales, group_size, pack_factor, bias=None):
    ...
    GROUP_SIZE_M = 8
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),   # 一维！
    )
    awq_gemm_kernel[grid](
        ...,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
```

#### 7.1.5 动手练习：验证分组调度的正确性（不涉及性能测量）

在真正接入 AWQ kernel 之前，建议先在**练习 3 的纯 fp16 GEMM kernel** 上做这个改动，因为逻辑更简单，方便你单独验证"改了调度顺序后，计算结果是否还是对的"（`GROUP_SIZE_M` 只改变 program 的**执行顺序**，不改变每个 program 算的**内容**，所以数值结果必须和不加这个优化时完全一致）：

```python
# 用练习 3 的 matmul，分别测试加了 GROUP_SIZE_M 前后的结果是否一致
a = torch.randn(256, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 384, device="cuda", dtype=torch.float16)

c_no_group = matmul(a, b)                       # 练习3原版
c_with_group = matmul_grouped(a, b)             # 你加了 GROUP_SIZE_M 之后的版本

assert torch.allclose(c_no_group, c_with_group, atol=1e-1, rtol=1e-2)
print("分组调度不改变计算结果，验证通过！")
```

**性能验证（选修，需要 GPU 上多次计时才能看出差异）**：

```python
import time

def bench(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000   # ms

# M、N、K 越大，GROUP_SIZE_M 的收益越明显（小矩阵可能看不出差异，甚至因调度开销略慢）
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
print("无分组:", bench(matmul, a, b), "ms")
print("有分组:", bench(matmul_grouped, a, b), "ms")
```

#### 7.1.6 这个技巧在 AWQ kernel 里怎么落地

对最终的 `awq_gemm_kernel`，做法完全一致：把 6.4 节骨架里 `pid_m = tl.program_id(0)` / `pid_n = tl.program_id(1)` 两行，换成 7.1.4 节的"一维 pid 反推分组坐标"的写法，再把 launch 函数的二维 `grid` 改成一维即可，**中间 for 循环里读 x/qweight/scales 的逻辑完全不用动**——这正是这个优化"低成本"的原因：只改了"外层怎么给 program 分配任务"，不改"每个 program 具体怎么算"。

因为 decode 阶段 `M` 通常很小（甚至 `M=1`），`num_pid_m` 本身就很小，`GROUP_SIZE_M` 的收益在 decode 场景不一定明显；但如果同时兼顾 prefill（`M` 较大）场景，这个优化就有意义，可以作为 Phase 3 完成基础功能后的**加分项**再补上，不阻塞主线交付。

---

## 8. 学习检查清单

完成本文档后，你应该能回答以下问题（自我检验）：

- [ ] `program_id`、`grid`、`BLOCK_SIZE` 三者是什么关系？为什么需要 `mask`？
- [ ] 为什么二维 tile 定位要用 `row[:, None]` 和 `col[None, :]` 这种广播写法？
- [ ] GEMM 里为什么要对 K 维做循环？`accumulator` 为什么用 fp32？
- [ ] int32 怎么在 Triton 里解包出多个 int4？位运算符怎么用？为什么要用二维广播一次性算出全部 nibble，而不是 for 循环里多次 store？
- [ ] 为什么 AWQ fused kernel 里建议 `BLOCK_K = group_size`？如果不这样设定会有什么麻烦？
- [ ] 你自己写出的 `awq_gemm_kernel` 能通过第 6.5 节的数值验证测试吗？
- [ ] `GROUP_SIZE_M` 为什么能提升 L2 Cache 命中率？它改变了 program 的什么，又没有改变什么？

如果最后一项做到了，说明你已经具备了独立扩展这个 kernel（支持 prefill 大 M、支持 autotune 调优）的能力，可以进入方案文档里的 Phase 3 验收阶段（decode 吞吐 / 显存对比 benchmark）。

---

## 9. 参考资料

- [Triton 官方 Vector Add 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton 官方 Matrix Multiplication 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)（练习 3、6 的原型来源）
- AutoAWQ Triton kernel 参考实现：`awq/modules/triton/gemm.py`（casper-hansen/AutoAWQ 仓库）
- 论文《Accelerating a Triton Fused Kernel for W4A16 Quantized Inference》（arXiv:2402.00025）—— 本项目 Phase B 设计思路来源
- 项目内已有 Triton 代码：`nanokvllm/layers/attention.py::store_kvcache_kernel`（最简单的项目内实例，建议对照着练习 1 再读一遍）
