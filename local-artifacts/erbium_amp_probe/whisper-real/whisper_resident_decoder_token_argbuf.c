/*
 * Resident raw-INT8 Whisper decoder-token executor.
 *
 * This is the first full decoder-side graph executor: token embedding,
 * positional embedding, all four decoder blocks, final LayerNorm, full vocab
 * projection, and greedy token selection run inside one ET-SoC1 kernel.
 *
 * Scope: decoder only.  The host supplies encoder cross-attention K/V caches
 * as FP16 in the 16 MiB runtime arena.  The decoder weights are read from the
 * 64 MiB resident raw-INT8 weight region.
 */

#include <stdint.h>

#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "whisper_resident_decoder_weights_auto.h"

extern char heap0_end[];

#ifndef MAGIC
#define MAGIC 0x57524443u
#endif
#ifndef RUNTIME_REGION_BYTES
#define RUNTIME_REGION_BYTES (16u * 1024u * 1024u)
#endif
#ifndef WEIGHT_REGION_OFFSET
#define WEIGHT_REGION_OFFSET (16u * 1024u * 1024u)
#endif

#define LAYERS 4u
#define HEADS 6u
#define HEAD_DIM 64u
#define DIM 384u
#define MLP_DIM 1536u
#define VOCAB 51864u
#define SRC_LEN 1500u
#define MAX_TOKENS 224u

#define SUMMARY_OFFSET 0x1000u
#define PARAM_OFFSET   0x2000u
#define TOKENS_OFFSET  0x3000u
#define HIDDEN0_OFFSET 0x4000u
#define HIDDEN1_OFFSET 0x5000u
#define QKV_OFFSET     0x7000u
#define TMP384_OFFSET  0xc000u
#define TMP1536_OFFSET 0x12000u
#define SCORES_OFFSET  0x2a000u
#define CONTEXT_OFFSET 0x31000u
#define CROSS_K_OFFSET 0x100000u
#define CROSS_V_OFFSET 0x580000u
#define SELF_K_OFFSET  0xa00000u
#define SELF_V_OFFSET  0xb80000u

struct run_params {
	uint32_t magic;
	uint32_t max_new_tokens;
	uint32_t reserved[14];
};

struct summary {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t max_new_tokens;
	uint32_t total_steps;
	uint32_t generated_tokens;
	uint32_t eos_seen;
	uint32_t final_token;
	uint32_t final_argmax;
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
	uint32_t reserved[9];
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

static float half_to_float(uint16_t h)
{
	const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
	const uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
	const uint32_t mant = (uint32_t)h & 0x3ffu;
	union {
		uint32_t u;
		float f;
	} out;

	if (exp == 0u) {
		if (mant == 0u) {
			out.u = sign;
			return out.f;
		}
		float v = (float)mant * (1.0f / 16777216.0f);
		return sign ? -v : v;
	}
	if (exp == 31u) {
		out.u = sign | 0x7f800000u | (mant << 13);
		return out.f;
	}
	out.u = sign | ((exp + 112u) << 23) | (mant << 13);
	return out.f;
}

static inline const int8_t *wptr(uint8_t *base, uint32_t off)
{
	return (const int8_t *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline float wval(uint8_t *base, struct weight_desc w, uint32_t idx)
{
	return (float)wptr(base, w.offset)[idx] * w.scale;
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

static void add_inplace(float *dst, const float *src)
{
	for (uint32_t i = 0u; i < DIM; i++) {
		dst[i] += src[i];
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

static inline uint32_t self_k_index(uint32_t layer, uint32_t head,
				    uint32_t dim, uint32_t step)
{
	return (((layer * HEADS + head) * HEAD_DIM + dim) * MAX_TOKENS) + step;
}

static inline uint32_t self_v_index(uint32_t layer, uint32_t head,
				    uint32_t step, uint32_t dim)
{
	return (((layer * HEADS + head) * MAX_TOKENS + step) * HEAD_DIM) + dim;
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

static void self_attention(uint8_t *base, uint32_t layer, uint32_t step,
			   float *hidden, float *ln, float *qkv,
			   float *context, float *scores, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];
	float *const self_k = (float *)(base + SELF_K_OFFSET);
	float *const self_v = (float *)(base + SELF_V_OFFSET);

	layernorm(base, ln, hidden, d.attn_ln_weight, d.attn_ln_bias);
	matvec_i8_nobias(base, qkv, ln, DIM, 1152u, d.self_qkv_weight);

	for (uint32_t i = 0u; i < DIM; i++) {
		qkv[i] += wval(base, d.self_q_bias, i);
		qkv[768u + i] += wval(base, d.self_v_bias, i);
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			self_k[self_k_index(layer, h, i, step)] =
				qkv[DIM + h * HEAD_DIM + i];
			self_v[self_v_index(layer, h, step, i)] =
				qkv[768u + h * HEAD_DIM + i];
		}
	}

	for (uint32_t i = 0u; i < DIM; i++) {
		context[i] = 0.0f;
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		const float *const q = qkv + h * HEAD_DIM;

		for (uint32_t t = 0u; t <= step; t++) {
			float s = 0.0f;

			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * self_k[self_k_index(layer, h, i, t)];
			}
			scores[t] = s;
		}
		softmax(scores, step + 1u);
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			float v = 0.0f;

			for (uint32_t t = 0u; t <= step; t++) {
				v += scores[t] * self_v[self_v_index(layer, h, t, i)];
			}
			context[h * HEAD_DIM + i] = v;
		}
	}

	matvec_i8(base, tmp, context, DIM, DIM, d.self_out_weight, d.self_out_bias);
	add_inplace(hidden, tmp);
}

static void cross_attention(uint8_t *base, uint32_t layer, float *hidden,
			    float *ln, float *query, float *context,
			    float *scores, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];
	const uint16_t *const cross_k = (const uint16_t *)(base + CROSS_K_OFFSET);
	const uint16_t *const cross_v = (const uint16_t *)(base + CROSS_V_OFFSET);

	layernorm(base, ln, hidden, d.cross_ln_weight, d.cross_ln_bias);
	matvec_i8(base, query, ln, DIM, DIM, d.cross_q_weight, d.cross_q_bias);

	for (uint32_t i = 0u; i < DIM; i++) {
		context[i] = 0.0f;
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		const float *const q = query + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			float s = 0.0f;

			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * half_to_float(
					cross_k[cross_k_index(layer, h, i, t)]);
			}
			scores[t] = s;
		}
		softmax(scores, SRC_LEN);
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			float v = 0.0f;

			for (uint32_t t = 0u; t < SRC_LEN; t++) {
				v += scores[t] * half_to_float(
					cross_v[cross_v_index(layer, h, t, i)]);
			}
			context[h * HEAD_DIM + i] = v;
		}
	}

	matvec_i8(base, tmp, context, DIM, DIM, d.cross_out_weight,
		  d.cross_out_bias);
	add_inplace(hidden, tmp);
}

static void mlp(uint8_t *base, uint32_t layer, float *hidden, float *ln,
		float *wide, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];

	layernorm(base, ln, hidden, d.mlp_ln_weight, d.mlp_ln_bias);
	matvec_i8(base, wide, ln, DIM, MLP_DIM, d.mlp_fc1_weight,
		  d.mlp_fc1_bias);

	for (uint32_t i = 0u; i < MLP_DIM; i++) {
		const float x = wide[i];
		const float y = 0.5f * x *
			(1.0f + fast_erff(x * 0.70710678118654757f));
		wide[i] = y;
	}

	matvec_i8(base, tmp, wide, MLP_DIM, DIM, d.mlp_fc2_weight,
		  d.mlp_fc2_bias);
	add_inplace(hidden, tmp);
}

static uint32_t final_argmax(uint8_t *base, float *hidden, float *ln)
{
	layernorm(base, ln, hidden, final_ln_weight, final_ln_bias);

	const int8_t *const wt = wptr(base, vocab_proj_weight.offset);
	float best = -3.4028234663852886e38f;
	uint32_t argmax = 0u;

	for (uint32_t n = 0u; n < VOCAB; n++) {
		float acc = 0.0f;

		for (uint32_t k = 0u; k < DIM; k++) {
			acc += ln[k] * ((float)wt[k * VOCAB + n] *
					vocab_proj_weight.scale);
		}
		if (acc > best || (acc == best && n < argmax)) {
			best = acc;
			argmax = n;
		}
	}

	return argmax;
}

static void run_step(uint8_t *base, uint32_t token, uint32_t step,
		     float *hidden, float *ln, float *qkv, float *tmp,
		     float *wide, float *scores, float *context)
{
	for (uint32_t i = 0u; i < DIM; i++) {
		hidden[i] = wval(base, token_embedding, token * DIM + i) +
			    wval(base, positional_embedding, step * DIM + i);
	}

	for (uint32_t layer = 0u; layer < LAYERS; layer++) {
		self_attention(base, layer, step, hidden, ln, qkv, context,
			       scores, tmp);
		cross_attention(base, layer, hidden, ln, tmp, context, scores,
				qkv);
		mlp(base, layer, hidden, ln, wide, tmp);
	}
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id != 0u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	volatile struct run_params *const params =
		(volatile struct run_params *)(base + PARAM_OFFSET);
	volatile struct summary *const s =
		(volatile struct summary *)(base + SUMMARY_OFFSET);
	uint32_t *const tokens = (uint32_t *)(base + TOKENS_OFFSET);
	float *const hidden = (float *)(base + HIDDEN0_OFFSET);
	float *const ln = (float *)(base + HIDDEN1_OFFSET);
	float *const qkv = (float *)(base + QKV_OFFSET);
	float *const tmp = (float *)(base + TMP384_OFFSET);
	float *const wide = (float *)(base + TMP1536_OFFSET);
	float *const scores = (float *)(base + SCORES_OFFSET);
	float *const context = (float *)(base + CONTEXT_OFFSET);
	uint32_t max_new = params->max_new_tokens;

	if (max_new > 64u) {
		max_new = 64u;
	}

	tokens[0] = 50257u;

	const uint64_t hpm3_0 = read_hpm3();
	const uint64_t hpm4_0 = read_hpm4();
	const uint64_t hpm5_0 = read_hpm5();
	const uint64_t hpm6_0 = read_hpm6();
	const uint64_t hpm7_0 = read_hpm7();
	const uint64_t hpm8_0 = read_hpm8();

	uint32_t total_steps = max_new + 3u;
	uint32_t eos_seen = 0u;
	uint32_t final_arg = 0u;
	uint32_t step;

	for (step = 0u; step < total_steps && step < (MAX_TOKENS - 1u); step++) {
		run_step(base, tokens[step], step, hidden, ln, qkv, tmp, wide,
			 scores, context);
		final_arg = final_argmax(base, hidden, ln);

		if (step == 0u) {
			tokens[step + 1u] = 50258u;
		} else if (step == 1u) {
			tokens[step + 1u] = 50358u;
		} else if (step == 2u) {
			tokens[step + 1u] = 50362u;
		} else {
			tokens[step + 1u] = final_arg;
			if (final_arg == 50256u) {
				eos_seen = 1u;
				step++;
				break;
			}
		}
	}

	const uint64_t hpm3_1 = read_hpm3();
	const uint64_t hpm4_1 = read_hpm4();
	const uint64_t hpm5_1 = read_hpm5();
	const uint64_t hpm6_1 = read_hpm6();
	const uint64_t hpm7_1 = read_hpm7();
	const uint64_t hpm8_1 = read_hpm8();

	const uint64_t ops_per_step =
		(uint64_t)LAYERS *
		((uint64_t)DIM * 1152u * 2u +
		 (uint64_t)DIM * DIM * 2u * 4u +
		 (uint64_t)DIM * MLP_DIM * 2u * 2u +
		 (uint64_t)HEADS * SRC_LEN * HEAD_DIM * 2u * 2u) +
		(uint64_t)DIM * VOCAB * 2u;
	const uint64_t ops = ops_per_step * step;

	s->magic = MAGIC;
	s->hart_id = hart_id;
	s->max_new_tokens = max_new;
	s->total_steps = step;
	s->generated_tokens = step > 3u ? step - 3u : 0u;
	s->eos_seen = eos_seen;
	s->final_token = tokens[step];
	s->final_argmax = final_arg;
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
	evict((void *)tokens, (step + 1u) * sizeof(uint32_t));
	evict((void *)s, sizeof(*s));
	WAIT_CACHEOPS;

	return 0;
}
