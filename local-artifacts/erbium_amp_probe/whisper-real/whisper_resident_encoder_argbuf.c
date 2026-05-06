/*
 * Resident raw-INT8 Whisper encoder executor.
 *
 * Input:  log-mel features at AUDIO_OFFSET in the 16 MiB runtime arena.
 * Output: decoder cross-attention K/V caches as FP16 at the same offsets used
 * by whisper_resident_decoder_token_argbuf.c.
 *
 * The packed raw-INT8 weight region starts at WEIGHT_REGION_OFFSET.  Temporary
 * hidden/qkv tensors live in the unused tail of that 64 MiB region, at the
 * generated ENCODER_SCRATCH_OFFSET.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "whisper_resident_encoder_weights_auto.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef MAGIC
#define MAGIC 0x5752454eu
#endif
#ifndef RUNTIME_REGION_BYTES
#define RUNTIME_REGION_BYTES (16u * 1024u * 1024u)
#endif
#ifndef WEIGHT_REGION_OFFSET
#define WEIGHT_REGION_OFFSET (16u * 1024u * 1024u)
#endif

#define HEADS 6u
#define HEAD_DIM 64u
#define DIM 384u
#define MLP_DIM 1536u
#define MEL_BINS 80u
#define AUDIO_LEN 3000u
#define SRC_LEN 1500u
#define QKV_DIM 1152u
#define LAYERS 4u

#define BENCH_FLB 2u
#define BENCH_FCC FCC_0

#define SUMMARY_OFFSET 0x1000u
#define PARAM_OFFSET   0x2000u
#define AUDIO_OFFSET   0x4000u
#define CROSS_K_OFFSET 0x100000u
#define CROSS_V_OFFSET 0x580000u
#define CONV1_OFFSET   0x100000u
#define SCRATCH_OFFSET 0xa00000u

#define ROW_LN_FLOATS      DIM
#define ROW_CONTEXT_FLOATS DIM
#define ROW_TMP_FLOATS     DIM
#define ROW_WIDE_FLOATS    MLP_DIM
#define ROW_SCORES_FLOATS  SRC_LEN
#define ROW_SLOT_FLOATS    (ROW_LN_FLOATS + ROW_CONTEXT_FLOATS + \
			    ROW_TMP_FLOATS + ROW_WIDE_FLOATS + \
			    ROW_SCORES_FLOATS)

struct run_params {
	uint32_t magic;
	uint32_t max_new_tokens;
	uint32_t reserved[14];
};

struct summary {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t active_harts;
	uint32_t src_len;
	uint32_t scratch_offset_lo;
	uint32_t scratch_offset_hi;
	uint32_t hpm3_lo;
	uint32_t hpm3_hi;
	uint32_t hpm4_lo;
	uint32_t hpm4_hi;
	uint32_t hpm5_lo;
	uint32_t hpm5_hi;
	uint32_t hpm6_lo;
	uint32_t hpm6_hi;
	uint32_t hpm7_lo;
	uint32_t hpm7_hi;
	uint32_t hpm8_lo;
	uint32_t hpm8_hi;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t done;
	uint32_t reserved[11];
};

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - RUNTIME_REGION_BYTES;
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - RUNTIME_REGION_BYTES;
	}

	return ptr;
}

static inline uint32_t active_mask_t0(void)
{
	uint32_t mask = 0u;

	for (uint32_t h = 0u; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1u);
	}

	return mask;
}

static inline uint32_t active_mask_t1(void)
{
	uint32_t mask = 0u;

	for (uint32_t h = 1u; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1u);
	}

	return mask;
}

static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
	}
}

static inline uint64_t read_hpm3(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter3" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm4(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter4" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm5(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter5" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm6(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter6" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm7(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter7" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm8(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter8" : "=r"(v));
	return v;
}

static inline float abs_f32(float v)
{
	return v < 0.0f ? -v : v;
}

static inline float make_pow2(int32_t n)
{
	union {
		uint32_t u;
		float f;
	} bits;

	if (n < -126) {
		return 0.0f;
	}
	if (n > 127) {
		bits.u = 0x7f800000u;
		return bits.f;
	}

	bits.u = (uint32_t)(n + 127) << 23;
	return bits.f;
}

static float fast_recipf(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	if (x == 0.0f) {
		return 3.4028234663852886e38f;
	}

	v.f = x;
	v.u = 0x7EF311C3u - v.u;
	v.f = v.f * (2.0f - x * v.f);
	v.f = v.f * (2.0f - x * v.f);
	v.f = v.f * (2.0f - x * v.f);
	return v.f;
}

static float fast_expf(float x)
{
	if (x <= -87.33654475f) {
		return 0.0f;
	}
	if (x >= 88.72283905f) {
		return 3.4028234663852886e38f;
	}

	const float inv_ln2 = 1.44269504088896341f;
	const float ln2_hi = 0.693359375f;
	const float ln2_lo = -0.00021219444005469057f;
	const float z = x * inv_ln2;
	const int32_t n = (int32_t)(z + (z >= 0.0f ? 0.5f : -0.5f));
	const float r = (x - ((float)n * ln2_hi)) - ((float)n * ln2_lo);
	const float r2 = r * r;
	const float r3 = r2 * r;
	const float r4 = r2 * r2;
	const float r5 = r4 * r;
	const float p = 1.0f + r + (0.5f * r2) + (0.1666666716f * r3) +
			(0.0416666679f * r4) + (0.0083333310f * r5);

	return p * make_pow2(n);
}

static float fast_erff(float x)
{
	const float ax = abs_f32(x);
	const float t = fast_recipf(1.0f + 0.3275911f * ax);
	const float a1 = 0.254829592f;
	const float a2 = -0.284496736f;
	const float a3 = 1.421413741f;
	const float a4 = -1.453152027f;
	const float a5 = 1.061405429f;
	const float poly =
		(((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
	const float y = 1.0f - poly * fast_expf(-(ax * ax));

	return x < 0.0f ? -y : y;
}

static float fast_inv_sqrtf(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	v.f = x;
	v.u = 0x5f3759dfu - (v.u >> 1u);
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	return v.f;
}

static uint16_t float_to_half(float f)
{
	union {
		float f;
		uint32_t u;
	} v;
	uint32_t sign;
	int32_t exp;
	uint32_t mant;

	v.f = f;
	sign = (v.u >> 16) & 0x8000u;
	exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
	mant = v.u & 0x7fffffu;

	if (exp <= 0) {
		if (exp < -10) {
			return (uint16_t)sign;
		}
		mant |= 0x800000u;
		const uint32_t shift = (uint32_t)(14 - exp);
		uint32_t hmant = mant >> shift;
		if ((mant >> (shift - 1u)) & 1u) {
			hmant++;
		}
		return (uint16_t)(sign | hmant);
	}
	if (exp >= 31) {
		return (uint16_t)(sign | 0x7c00u);
	}

	mant += 0x1000u;
	if (mant & 0x800000u) {
		mant = 0u;
		exp++;
	}
	if (exp >= 31) {
		return (uint16_t)(sign | 0x7c00u);
	}
	return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static inline const int8_t *wptr(uint8_t *base, uint32_t off)
{
	return (const int8_t *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline float wval(uint8_t *base, struct weight_desc w, uint32_t idx)
{
	return (float)wptr(base, w.offset)[idx] * w.scale;
}

static inline float gelu(float x)
{
	return 0.5f * x * (1.0f + fast_erff(x * 0.70710678118654757f));
}

static void layernorm(uint8_t *base, float *out, const float *in,
		      struct weight_desc gamma, struct weight_desc beta)
{
	float sum = 0.0f;
	float sumsq = 0.0f;

	for (uint32_t i = 0u; i < DIM; i++) {
		const float v = in[i];
		sum += v;
		sumsq += v * v;
	}

	const float mean = sum * (1.0f / (float)DIM);
	const float var = (sumsq * (1.0f / (float)DIM)) - (mean * mean);
	const float inv = fast_inv_sqrtf(var + 0.00001f);

	for (uint32_t i = 0u; i < DIM; i++) {
		out[i] = ((in[i] - mean) * inv) * wval(base, gamma, i) +
			 wval(base, beta, i);
	}
}

static void matvec_i8(uint8_t *base, float *out, const float *in,
		      uint32_t in_dim, uint32_t out_dim,
		      struct weight_desc weight, struct weight_desc bias)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);

	for (uint32_t n = 0u; n < out_dim; n++) {
		float acc = (float)b[n] * bias.scale;

		for (uint32_t k = 0u; k < in_dim; k++) {
			acc += in[k] * ((float)wt[k * out_dim + n] * weight.scale);
		}
		out[n] = acc;
	}
}

static void matvec_i8_nobias(uint8_t *base, float *out, const float *in,
			     uint32_t in_dim, uint32_t out_dim,
			     struct weight_desc weight)
{
	const int8_t *const wt = wptr(base, weight.offset);

	for (uint32_t n = 0u; n < out_dim; n++) {
		float acc = 0.0f;

		for (uint32_t k = 0u; k < in_dim; k++) {
			acc += in[k] * ((float)wt[k * out_dim + n] * weight.scale);
		}
		out[n] = acc;
	}
}

static void softmax(float *scores, uint32_t count)
{
	float maxv = -3.4028234663852886e38f;
	float sum = 0.0f;

	for (uint32_t i = 0u; i < count; i++) {
		if (scores[i] > maxv) {
			maxv = scores[i];
		}
	}
	for (uint32_t i = 0u; i < count; i++) {
		const float e = fast_expf(scores[i] - maxv);
		scores[i] = e;
		sum += e;
	}
	const float inv = fast_recipf(sum);
	for (uint32_t i = 0u; i < count; i++) {
		scores[i] *= inv;
	}
}

static inline uint32_t part0(uint32_t n, uint32_t hart_id)
{
	return (n * hart_id) / ACTIVE_HARTS;
}

static inline uint32_t part1(uint32_t n, uint32_t hart_id)
{
	return (n * (hart_id + 1u)) / ACTIVE_HARTS;
}

static inline uint32_t cross_k_index(uint32_t layer, uint32_t head,
				     uint32_t dim, uint32_t pos)
{
	return (((layer * HEADS + head) * HEAD_DIM + dim) * SRC_LEN) + pos;
}

static inline uint32_t cross_v_index(uint32_t layer, uint32_t head,
				     uint32_t pos, uint32_t dim)
{
	return (((layer * HEADS + head) * SRC_LEN + pos) * HEAD_DIM) + dim;
}

static void conv1_stage(uint8_t *base, uint32_t hart_id, float *audio,
			float *conv1)
{
	const uint32_t total = DIM * AUDIO_LEN;
	const uint32_t i0 = part0(total, hart_id);
	const uint32_t i1 = part1(total, hart_id);

	for (uint32_t idx = i0; idx < i1; idx++) {
		const uint32_t oc = idx / AUDIO_LEN;
		const uint32_t t = idx - oc * AUDIO_LEN;
		float acc = wval(base, enc_conv1_bias, oc);

		for (uint32_t ic = 0u; ic < MEL_BINS; ic++) {
			for (uint32_t k = 0u; k < 3u; k++) {
				const int32_t src_t = (int32_t)t + (int32_t)k - 1;

				if (src_t >= 0 && src_t < (int32_t)AUDIO_LEN) {
					const uint32_t widx =
						((oc * MEL_BINS + ic) * 3u) + k;
					acc += audio[ic * AUDIO_LEN + (uint32_t)src_t] *
					       wval(base, enc_conv1_weight, widx);
				}
			}
		}
		conv1[idx] = gelu(acc);
	}
}

static void conv2_stage(uint8_t *base, uint32_t hart_id, float *conv1,
			float *hidden)
{
	const uint32_t total = SRC_LEN * DIM;
	const uint32_t i0 = part0(total, hart_id);
	const uint32_t i1 = part1(total, hart_id);

	for (uint32_t idx = i0; idx < i1; idx++) {
		const uint32_t t = idx / DIM;
		const uint32_t oc = idx - t * DIM;
		float acc = wval(base, enc_conv2_bias, oc);

		for (uint32_t ic = 0u; ic < DIM; ic++) {
			for (uint32_t k = 0u; k < 3u; k++) {
				const int32_t src_t =
					(int32_t)(t * 2u) + (int32_t)k - 1;

				if (src_t >= 0 && src_t < (int32_t)AUDIO_LEN) {
					const uint32_t widx =
						((oc * DIM + ic) * 3u) + k;
					acc += conv1[ic * AUDIO_LEN + (uint32_t)src_t] *
					       wval(base, enc_conv2_weight, widx);
				}
			}
		}
		hidden[t * DIM + oc] = gelu(acc) +
				       wval(base, enc_positional_embedding, idx);
	}
}

static void qkv_stage(uint8_t *base, uint32_t hart_id, uint32_t layer,
		      float *hidden, float *qkv, float *ln)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);

	for (uint32_t row = r0; row < r1; row++) {
		float *const dst = qkv + row * QKV_DIM;

		layernorm(base, ln, hidden + row * DIM, d.attn_ln_weight,
			  d.attn_ln_bias);
		matvec_i8_nobias(base, dst, ln, DIM, QKV_DIM, d.self_qkv_weight);
		for (uint32_t i = 0u; i < DIM; i++) {
			dst[i] += wval(base, d.self_q_bias, i);
			dst[768u + i] += wval(base, d.self_v_bias, i);
		}
	}
}

static void attention_stage(uint8_t *base, uint32_t hart_id, uint32_t layer,
			    float *hidden, float *qkv, float *context,
			    float *scores, float *tmp)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);
	const float scale = wval(base, enc_attn_scale, 0u);
	const float score_scale = scale * scale;

	for (uint32_t row = r0; row < r1; row++) {
		for (uint32_t i = 0u; i < DIM; i++) {
			context[i] = 0.0f;
		}

		for (uint32_t h = 0u; h < HEADS; h++) {
			const float *const q = qkv + row * QKV_DIM + h * HEAD_DIM;

			for (uint32_t t = 0u; t < SRC_LEN; t++) {
				const float *const k =
					qkv + t * QKV_DIM + DIM + h * HEAD_DIM;
				float s = 0.0f;

				for (uint32_t i = 0u; i < HEAD_DIM; i++) {
					s += q[i] * k[i];
				}
				scores[t] = s * score_scale;
			}
			softmax(scores, SRC_LEN);
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				float v = 0.0f;

				for (uint32_t t = 0u; t < SRC_LEN; t++) {
					v += scores[t] *
					     qkv[t * QKV_DIM + 768u + h * HEAD_DIM + i];
				}
				context[h * HEAD_DIM + i] = v;
			}
		}

		matvec_i8(base, tmp, context, DIM, DIM, d.self_out_weight,
			  d.self_out_bias);
		for (uint32_t i = 0u; i < DIM; i++) {
			hidden[row * DIM + i] += tmp[i];
		}
	}
}

static void mlp_stage(uint8_t *base, uint32_t hart_id, uint32_t layer,
		      float *hidden, float *ln, float *wide, float *tmp)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);

	for (uint32_t row = r0; row < r1; row++) {
		layernorm(base, ln, hidden + row * DIM, d.mlp_ln_weight,
			  d.mlp_ln_bias);
		matvec_i8(base, wide, ln, DIM, MLP_DIM, d.mlp_fc1_weight,
			  d.mlp_fc1_bias);
		for (uint32_t i = 0u; i < MLP_DIM; i++) {
			wide[i] = gelu(wide[i]);
		}
		matvec_i8(base, tmp, wide, MLP_DIM, DIM, d.mlp_fc2_weight,
			  d.mlp_fc2_bias);
		for (uint32_t i = 0u; i < DIM; i++) {
			hidden[row * DIM + i] += tmp[i];
		}
	}
}

static void cross_cache_stage(uint8_t *base, uint32_t hart_id, float *hidden,
			      float *ln)
{
	uint16_t *const cross_k = (uint16_t *)(base + CROSS_K_OFFSET);
	uint16_t *const cross_v = (uint16_t *)(base + CROSS_V_OFFSET);
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);

	for (uint32_t row = r0; row < r1; row++) {
		layernorm(base, ln, hidden + row * DIM, enc_ln_post_weight,
			  enc_ln_post_bias);

		for (uint32_t layer = 0u; layer < LAYERS; layer++) {
			for (uint32_t h = 0u; h < HEADS; h++) {
				const struct weight_desc kw =
					enc_cross_k_weight[layer][h];
				const struct weight_desc vw =
					enc_cross_v_weight[layer][h];
				const struct weight_desc vb =
					enc_cross_v_bias[layer][h];

				for (uint32_t d = 0u; d < HEAD_DIM; d++) {
					float kacc = 0.0f;
					float vacc = wval(base, vb, d);

					for (uint32_t i = 0u; i < DIM; i++) {
						kacc += ln[i] * wval(base, kw,
								     i * HEAD_DIM + d);
						vacc += ln[i] * wval(base, vw,
								     i * HEAD_DIM + d);
					}
					cross_k[cross_k_index(layer, h, d, row)] =
						float_to_half(kacc);
					cross_v[cross_v_index(layer, h, row, d)] =
						float_to_half(vacc);
				}
			}
		}
	}
}

static void flush_all_encoder_state(float *hidden, float *qkv)
{
	FENCE;
	evict(hidden, SRC_LEN * DIM * sizeof(float));
	evict(qkv, SRC_LEN * QKV_DIM * sizeof(float));
	WAIT_CACHEOPS;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	volatile struct summary *const s =
		(volatile struct summary *)(base + SUMMARY_OFFSET);
	float *const audio = (float *)(base + AUDIO_OFFSET);
	float *const conv1 = (float *)(base + CONV1_OFFSET);
	float *const hidden = (float *)(base + WEIGHT_REGION_OFFSET +
					ENCODER_SCRATCH_OFFSET);
	float *const qkv = hidden + (SRC_LEN * DIM);
	float *const slot = (float *)(base + SCRATCH_OFFSET +
				      hart_id * ROW_SLOT_FLOATS * sizeof(float));
	float *const ln = slot;
	float *const context = ln + ROW_LN_FLOATS;
	float *const tmp = context + ROW_CONTEXT_FLOATS;
	float *const wide = tmp + ROW_TMP_FLOATS;
	float *const scores = wide + ROW_WIDE_FLOATS;

	if (hart_id == 0u) {
		FENCE;
		evict(audio, MEL_BINS * AUDIO_LEN * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint64_t hpm3_0 = read_hpm3();
	const uint64_t hpm4_0 = read_hpm4();
	const uint64_t hpm5_0 = read_hpm5();
	const uint64_t hpm6_0 = read_hpm6();
	const uint64_t hpm7_0 = read_hpm7();
	const uint64_t hpm8_0 = read_hpm8();

	conv1_stage(base, hart_id, audio, conv1);
	FENCE;
	evict(conv1, DIM * AUDIO_LEN * sizeof(float));
	WAIT_CACHEOPS;
	bench_barrier();

	conv2_stage(base, hart_id, conv1, hidden);
	FENCE;
	evict(hidden, SRC_LEN * DIM * sizeof(float));
	WAIT_CACHEOPS;
	bench_barrier();

	for (uint32_t layer = 0u; layer < LAYERS; layer++) {
		qkv_stage(base, hart_id, layer, hidden, qkv, ln);
		FENCE;
		evict(qkv, SRC_LEN * QKV_DIM * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();

		attention_stage(base, hart_id, layer, hidden, qkv, context,
				scores, tmp);
		FENCE;
		evict(hidden, SRC_LEN * DIM * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();

		mlp_stage(base, hart_id, layer, hidden, ln, wide, tmp);
		FENCE;
		evict(hidden, SRC_LEN * DIM * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();
	}

	cross_cache_stage(base, hart_id, hidden, ln);
	FENCE;
	evict((void *)(base + CROSS_K_OFFSET),
	      LAYERS * HEADS * HEAD_DIM * SRC_LEN * sizeof(uint16_t));
	evict((void *)(base + CROSS_V_OFFSET),
	      LAYERS * HEADS * SRC_LEN * HEAD_DIM * sizeof(uint16_t));
	WAIT_CACHEOPS;
	bench_barrier();

	const uint64_t hpm3_1 = read_hpm3();
	const uint64_t hpm4_1 = read_hpm4();
	const uint64_t hpm5_1 = read_hpm5();
	const uint64_t hpm6_1 = read_hpm6();
	const uint64_t hpm7_1 = read_hpm7();
	const uint64_t hpm8_1 = read_hpm8();

	if (hart_id == 0u) {
		const uint64_t conv_ops =
			((uint64_t)DIM * AUDIO_LEN * MEL_BINS * 3u * 2u) +
			((uint64_t)DIM * SRC_LEN * DIM * 3u * 2u);
		const uint64_t block_ops = (uint64_t)LAYERS *
			(((uint64_t)SRC_LEN * DIM * QKV_DIM * 2u) +
			 ((uint64_t)SRC_LEN * HEADS * SRC_LEN * HEAD_DIM * 2u) +
			 ((uint64_t)SRC_LEN * HEADS * SRC_LEN * HEAD_DIM * 2u) +
			 ((uint64_t)SRC_LEN * DIM * DIM * 2u) +
			 ((uint64_t)SRC_LEN * DIM * MLP_DIM * 2u) +
			 ((uint64_t)SRC_LEN * MLP_DIM * DIM * 2u));
		const uint64_t cross_ops =
			(uint64_t)LAYERS * HEADS * SRC_LEN * DIM * HEAD_DIM * 2u * 2u;
		const uint64_t ops = conv_ops + block_ops + cross_ops;

		s->magic = MAGIC;
		s->hart_id = hart_id;
		s->active_harts = ACTIVE_HARTS;
		s->src_len = SRC_LEN;
		s->scratch_offset_lo = (uint32_t)ENCODER_SCRATCH_OFFSET;
		s->scratch_offset_hi = (uint32_t)(ENCODER_SCRATCH_OFFSET >> 32);
		s->hpm3_lo = (uint32_t)(hpm3_1 - hpm3_0);
		s->hpm3_hi = (uint32_t)((hpm3_1 - hpm3_0) >> 32);
		s->hpm4_lo = (uint32_t)(hpm4_1 - hpm4_0);
		s->hpm4_hi = (uint32_t)((hpm4_1 - hpm4_0) >> 32);
		s->hpm5_lo = (uint32_t)(hpm5_1 - hpm5_0);
		s->hpm5_hi = (uint32_t)((hpm5_1 - hpm5_0) >> 32);
		s->hpm6_lo = (uint32_t)(hpm6_1 - hpm6_0);
		s->hpm6_hi = (uint32_t)((hpm6_1 - hpm6_0) >> 32);
		s->hpm7_lo = (uint32_t)(hpm7_1 - hpm7_0);
		s->hpm7_hi = (uint32_t)((hpm7_1 - hpm7_0) >> 32);
		s->hpm8_lo = (uint32_t)(hpm8_1 - hpm8_0);
		s->hpm8_hi = (uint32_t)((hpm8_1 - hpm8_0) >> 32);
		s->ops_lo = (uint32_t)ops;
		s->ops_hi = (uint32_t)(ops >> 32);
		s->done = 1u;
		FENCE;
		evict((void *)s, sizeof(*s));
		WAIT_CACHEOPS;
	}

	bench_barrier();
	(void)flush_all_encoder_state;
	return 0;
}
