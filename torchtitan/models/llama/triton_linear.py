import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch import nn
from .data_parallel_gemm import gemm_data_parallel_kernel



@triton.jit()
def tile_schedule(pid,
                  NUM_SM: tl.constexpr, total_tiles: tl.constexpr):

    start = (pid*total_tiles) // NUM_SM
    end = (((pid+1)*total_tiles) // NUM_SM)

    return start, end

@triton.jit()
def gemm_balanced(a_ptr, b_ptr, c_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            prob_m, prob_n, prob_k,
            block_m: tl.constexpr,
            block_n: tl.constexpr,
            block_k: tl.constexpr,
            NUM_SM: tl.constexpr,
            total_tiles: tl.constexpr,
            ):

    pid = tl.program_id(0)

    num_n_tiles = tl.cdiv(prob_n, block_n)
    num_k_tiles = tl.cdiv(prob_k, block_k)

    start, end = tile_schedule(pid, NUM_SM, total_tiles)
    for tile_id in range(start, end):

        tile_m_idx = tile_id // num_n_tiles
        tile_n_idx = tile_id % num_n_tiles

        offs_m = tile_m_idx*block_m + tl.arange(0, block_m)
        offs_n = tile_n_idx*block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        # Compiler Hint for Vectorized Load
        offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

        a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn)

        acc = tl.zeros([block_m, block_n], tl.float32)
        for kk in range(0, num_k_tiles):

            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

            a_ptrs += block_k*stride_ak
            b_ptrs += block_k*stride_bk

        acc.to(tl.bfloat16)

        offs_cm = tile_m_idx*block_m + tl.arange(0, block_m)
        offs_cn = tile_n_idx*block_n + tl.arange(0, block_n)

        c_ptrs = c_ptr + stride_cm*offs_cm[:, None] + stride_cn*offs_cn[None, :]
        tl.store(c_ptrs, acc)

@triton.jit
def gemm_grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # breakpoint()
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):

                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)

                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            c = accumulator.to(tl.bfloat16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(group_A, group_B, config = None):
    device = 'cuda'
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):

        A = group_A[i]
        B = group_B[i]
        # (f"{A.shape=}, {B.shape=}")
        #A.shape=torch.Size([16384, 4096]), B.shape=torch.Size([11008, 4096])
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable

    num_sm = 132
    '''if config:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]

    else:
    '''
    block_m = 64 # 128
    block_n = 64 # 256
    block_k = 64 # 32
    num_warps = 8
    num_stages = 4

    grid = (num_sm, )
    gemm_grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM=num_sm,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return group_C


@torch.library.custom_op("triton::matmul", mutates_args=())
def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    m, k = a.shape
    _, n = b.shape

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


@matmul_fn.register_fake
def _(a, b):
    m, _ = a.shape
    _, n = b.shape
    return a.new_empty(m, n)

class _matmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        #print(f"a pre squeeze shape: {a.shape}, {b.shape=}")
        #print(f"activation datatype {a.dtype=} and weight datatype {b.dtype}")
        if a.dtype == torch.float32:
            a = a.to(torch.bfloat16)
        if b.dtype == torch.float32:
            b = b.to(torch.bfloat16)

        #print(f"activation datatype after casting {a.dtype=} and weight datatype after casting {b.dtype}")
        s = a.size()
        if a.dim() >= 3:
            #a = torch.flatten(a, start_dim=-2)
            a = a.view(s[0] * s[1], s[2])

        assert a.dtype== b.dtype, f"mismatch: {a.dtype}, {b.dtype}"
        #print(f"a post squeeze shape: {a.shape}")
        m, k1 = a.shape # 16384, 11008
        k, n = b.shape # 4096, 11008
        if k1 != k:
            b = b.T  # 11008, 4096
            #print(f"b post transpose shape: {b.shape= }, {a.shape=}")
        #print(f"m post squeeze shape: {m=}")
        k, n = b.shape

        c = matmul_fn(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dL_dc):
        
        # print(f"line 314:  *** In backward final return *** ")
        """
        Equations:
                1. dL/da = dL/dc @ b.T
                2. dL/db = a.T @ dL/dc

        Shapes:
                GEMM1: 
                dL/dc: (m, n)
                b.T:   (n, k)
                ===> dL/da: (m, k)

                GEMM2:
                a.T:    (k, m)
                dL/dc:  (m, n)
                ===> dL/db: (k, n)

        Notation:
                dL/da: partial(L)/partial(a)
                dL/db: partial(L)/partial(b)
                dL/dc: partial(L)/partial(c)
        """
        a, b = ctx.saved_tensors
        # print(f"Input Shapes: Activations={a.shape=}, Weights={b.shape}, dL_dc={dL_dc.shape}")
        # print(f"Backward 1st GEMM {dL_dc.shape=} @ {b.T.shape=}")
        # print(f"Backward 2nd GEMM {a.T.shape=} @ {dL_dc.shape=}")
        

        dL_dc = dL_dc.to(torch.bfloat16)
        # test_dl_da = torch.matmul(dL_dc, b.T)
        # test_dl_db = torch.matmul(a.T, dL_dc)

        test_dl_da = matmul_fn(dL_dc, b.T)
        test_dl_db = matmul_fn(a.T, dL_dc)

        return test_dl_da.unsqueeze(0), test_dl_db

        group_a = [dL_dc, a.permute(1, 0).contiguous()]
        # print(f"***** past group_a..")
        group_b = [b.permute(1, 0).contiguous(), dL_dc]
        # print(f"**** past group_b..")

        group_derivs = group_gemm_fn(group_a, group_b)
        # print(f"**** past group_derivs..")
        # print(f"{group_derivs[0].shape=}, {group_derivs[1].shape=}")

        return group_derivs[0].unsqueeze(0), group_derivs[1]

matmul = _matmul.apply


# Define the custom linear layer
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TritonLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32, device='cuda'))


    def forward(self, input):

        c = matmul(input, self.weight.T)
        # print(f"{c.shape=}")
        # c = torch.matmul(input, self.weight.T)
        return c 