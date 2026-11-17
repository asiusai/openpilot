// SPDX-License-Identifier: MIT
// CK specialization derived from AMD's MIT-licensed generator at c56c6750d0fc54ed771d532cc92c316423449614.
// This wrapper supplies only the fixed B1/H16/KVH8/R2048/K2048/D128 causal BHSD contract.
#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp"

#if !defined(__gfx1200__) || !defined(__HIP_DEVICE_COMPILE__)
#error This translation unit requires device-only gfx1200 compilation.
#endif
static_assert(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT == 5);
static_assert(CK_TILE_FMHA_FWD_FAST_EXP2 == 1);

using BF16 = ck_tile::bf16_t;
using Shape = ck_tile::TileFmhaShape<ck_tile::sequence<64, 64, 32, 128, 32, 128>,
    ck_tile::sequence<4, 1, 1>, ck_tile::sequence<16, 16, 16>,
    ck_tile::sequence<4, 1, 1>, ck_tile::sequence<16, 16, 16>, true>;
using Traits = ck_tile::TileFmhaTraits<true, true, false, false, false,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, true, false,
    ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE, -1, false, false>;
using Variant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
using Mask = ck_tile::SimplifiedGenericAttentionMask<true>;
using Problem = ck_tile::BlockFmhaPipelineProblem<BF16, BF16, BF16, float, float, BF16,
    uint8_t, float, BF16, float, BF16, Shape, false, Variant, Mask, false, Traits>;
using Pipeline = ck_tile::BlockFmhaPipelineQRKSVS<Problem>;
using Epilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, BF16, true, false>>;
using Kernel = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
static_assert(sizeof(BF16) == 2);
static_assert(Kernel::kBlockSize == 128);
static_assert(Kernel::kBlockPerCu == 2);
static_assert(Kernel::GetSmemSize() == 9216);
static_assert(Kernel::kStoreLSE && !Kernel::kHasDropout && !Kernel::kIsGroupMode);

extern "C" __global__ __launch_bounds__(128, 2)
void alpamayo_ck_d128(BF16* output, float* lse, const BF16* query, const BF16* key, const BF16* value)
{
    Kernel::Kargs a{};
    a.q_ptr = query;
    a.k_ptr = key;
    a.v_ptr = value;
    a.o_ptr = output;
    a.sink_ptr = nullptr;
    a.seqlen_q = a.seqlen_k = 2048;
    a.hdim_q = a.hdim_v = 128;
    a.num_head_q = 16;
    a.nhead_ratio_qk = 2;
    constexpr float input_scale = 0.0883883461356163f;
    a.scale_s = static_cast<float>(input_scale * ck_tile::log2e_v<>);
    a.stride_q = a.stride_k = a.stride_v = a.stride_o = 128;
    a.nhead_stride_q = a.nhead_stride_k = a.nhead_stride_v = a.nhead_stride_o = 262144;
    a.batch_stride_q = a.batch_stride_o = 4194304;
    a.batch_stride_k = a.batch_stride_v = 2097152;
    a.num_head_q_total = 0;
    a.head_start = 0;
    a.window_size_left = -1;
    a.window_size_right = 0;
    a.sink_size = 0;
    a.mask_type = ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
    a.lse_ptr = lse;
    a.nhead_stride_lse = 2048;
    a.batch_stride_lse = 32768;
    a.cu_seqlen_q_ptr = a.cu_seqlen_k_ptr = nullptr;
    Kernel{}(a);
}
