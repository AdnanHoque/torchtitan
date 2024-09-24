import torch
import triton
import triton.language as tl


@triton.jit()
def column_major(pid,
              m, n,
              block_m: tl.constexpr, block_n: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m) 

    pid_m = pid % grid_m
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit()
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr):
    
    group_m = 16
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit
def gemm_data_parallel_kernel(a_ptr, b_ptr, c_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            m, n, k,
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
            ):
    
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_k = tl.cdiv(k, block_k)

    pid_m, pid_n = column_major(pid,
                                m, n,
                                block_m, block_n)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, num_pid_k):
        
        k_remaining = k - k_ * (block_k)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
    
    acc.to(tl.bfloat16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    tl.store(c_ptrs, acc, mask=mask)




class _matmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        m, _ = a.shape # 16384, 11008
        k, n = b.shape # 4096, 11008

        block_m = 128
        block_n = 256
        block_k = 64
        num_warps = 8
        num_stages = 4
        
        total_blocks_m = triton.cdiv(m, block_m)
        total_blocks_n = triton.cdiv(n, block_n)
        total_programs_mn = total_blocks_m * total_blocks_n
        
        grid = (total_programs_mn, 1, 1)
        c = torch.zeros((m, n), device=a.device, dtype=torch.bfloat16)
        k = gemm_data_parallel_kernel[grid](a, b, c,
                                a.stride(0), a.stride(1),
                                b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1),                        
                                m, n, k,
                                block_m, block_n, block_k,
                                num_warps=num_warps, num_stages=num_stages)
        
        return c

matmul_dp = _matmul.apply


if __name__ == '__main__':

    m, n, k = 2048, 4096, 4096
    a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')

    c = matmul_dp(a, b)
    d = torch.matmul(a, b)

    print(f"{c=}")
    print(f"{d=}")