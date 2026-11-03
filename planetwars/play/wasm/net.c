/* The value network's forward pass (entry/pwnet.py value_fast) in WebAssembly SIMD, single precision like the entry.

   No libc: memory is laid out by pw.js, which passes pointers. Weights are the `prepare`d arrays packed back to back:
     wPlanet [21*dm], wGlob [9*dm], bIn [dm],
     per layer: wqkv [dm*3dm], bqkv [3dm], wo [dm*dm], bo [dm], w1 [dm*ff], b1 [ff], w2 [ff*dm], b2 [dm],
     wPool [dm*dm], bG [dm], wGGlob [9*dm], v1w [dm*dm], v1b [dm], v2w [dm], v2b [1].
   dm, 3*dm and ff must be multiples of 8 and dm / heads a multiple of 4.

   Build: planetwars/play/wasm/build.sh */
#include <wasm_simd128.h>

#define NP 21
#define NG 9

static inline v128_t madd(v128_t c, v128_t a, v128_t b) { return wasm_f32x4_add(c, wasm_f32x4_mul(a, b)); }
static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

/* out[M x N] = A[M x K] @ W[K x N] (+ bias), N a multiple of 8. Four rows by eight columns at a time. */
static void matmul(const float *A, int M, int K, const float *W, int N, const float *bias, float *out) {
  int i = 0;
  for (; i + 4 <= M; i += 4) {
    const float *a0 = A + i * K, *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
    for (int j = 0; j < N; j += 8) {
      v128_t b0 = bias ? wasm_v128_load(bias + j) : wasm_f32x4_splat(0.0f);
      v128_t b1 = bias ? wasm_v128_load(bias + j + 4) : wasm_f32x4_splat(0.0f);
      v128_t c00 = b0, c01 = b1, c10 = b0, c11 = b1, c20 = b0, c21 = b1, c30 = b0, c31 = b1;
      const float *w = W + j;
      for (int k = 0; k < K; k++, w += N) {
        v128_t w0 = wasm_v128_load(w), w1 = wasm_v128_load(w + 4);
        v128_t x = wasm_f32x4_splat(a0[k]);
        c00 = madd(c00, x, w0); c01 = madd(c01, x, w1);
        x = wasm_f32x4_splat(a1[k]);
        c10 = madd(c10, x, w0); c11 = madd(c11, x, w1);
        x = wasm_f32x4_splat(a2[k]);
        c20 = madd(c20, x, w0); c21 = madd(c21, x, w1);
        x = wasm_f32x4_splat(a3[k]);
        c30 = madd(c30, x, w0); c31 = madd(c31, x, w1);
      }
      float *o = out + i * N + j;
      wasm_v128_store(o, c00); wasm_v128_store(o + 4, c01); o += N;
      wasm_v128_store(o, c10); wasm_v128_store(o + 4, c11); o += N;
      wasm_v128_store(o, c20); wasm_v128_store(o + 4, c21); o += N;
      wasm_v128_store(o, c30); wasm_v128_store(o + 4, c31);
    }
  }
  for (; i < M; i++) {
    const float *a = A + i * K;
    for (int j = 0; j < N; j += 8) {
      v128_t c0 = bias ? wasm_v128_load(bias + j) : wasm_f32x4_splat(0.0f);
      v128_t c1 = bias ? wasm_v128_load(bias + j + 4) : wasm_f32x4_splat(0.0f);
      const float *w = W + j;
      for (int k = 0; k < K; k++, w += N) {
        v128_t x = wasm_f32x4_splat(a[k]);
        c0 = madd(c0, x, wasm_v128_load(w)); c1 = madd(c1, x, wasm_v128_load(w + 4));
      }
      wasm_v128_store(out + i * N + j, c0); wasm_v128_store(out + i * N + j + 4, c1);
    }
  }
}

/* Layer norm of each row (D a multiple of 4), without gain and bias: those are folded into the next matrix. */
static void norm(const float *X, int rows, int D, float *out) {
  for (int r = 0; r < rows; r++) {
    const float *x = X + r * D;
    float *o = out + r * D;
    v128_t s = wasm_f32x4_splat(0.0f);
    for (int j = 0; j < D; j += 4) s = wasm_f32x4_add(s, wasm_v128_load(x + j));
    v128_t mu = wasm_f32x4_splat(hsum(s) / (float)D);
    v128_t v = wasm_f32x4_splat(0.0f);
    for (int j = 0; j < D; j += 4) {
      v128_t c = wasm_f32x4_sub(wasm_v128_load(x + j), mu);
      wasm_v128_store(o + j, c);
      v = madd(v, c, c);
    }
    v128_t inv = wasm_f32x4_splat(1.0f / __builtin_sqrtf(hsum(v) * (1.0f / (float)D) + 1e-5f));
    for (int j = 0; j < D; j += 4) wasm_v128_store(o + j, wasm_f32x4_mul(wasm_v128_load(o + j), inv));
  }
}

static const double FACT[9] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};

/* e^x for x <= 0, to float precision (worked in double: 2^integer part times a series for the rest). */
static float exp_neg(float xf) {
  double x = xf;
  if (x < -87.0) return 0.0f;
  double t = x * 1.4426950408889634;
  double fi = __builtin_floor(t);
  double r = (t - fi) * 0.6931471805599453;
  double p = 1.0 / 362880;  /* Horner form of the series for e^r, r in [0, ln 2) */
  for (int k = 8; k >= 0; k--) p = p * r + 1.0 / FACT[k];
  union { double d; long long i; } u;
  u.i = (long long)((int)fi + 1023) << 52;
  return (float)(p * u.d);
}

__attribute__((export_name("value_batch")))
void value_batch(const float *Wt, int dm, int ff, int heads, int layers, const float *planet, const float *glob, int B, int n,
                 float *out, float *scratch) {
  const int R = B * n, dh = dm / heads, d3 = 3 * dm;
  float *h = scratch, *a = h + R * dm, *qkv = a + R * dm, *o = qkv + R * d3, *u = o + R * dm, *t = u + R * ff;
  float *att = t + R * dm, *gv = att + ((n + 3) & ~3), *g2 = gv + dm, *pooled = g2 + dm;
  const float *wPlanet = Wt, *wGlob = wPlanet + NP * dm, *bIn = wGlob + NG * dm, *p = bIn + dm;

  matmul(planet, R, NP, wPlanet, dm, 0, h);
  for (int b = 0; b < B; b++) {
    matmul(glob + b * NG, 1, NG, wGlob, dm, bIn, gv);
    for (int q = 0; q < n; q++) {
      float *row = h + (b * n + q) * dm;
      for (int j = 0; j < dm; j += 4) wasm_v128_store(row + j, wasm_f32x4_add(wasm_v128_load(row + j), wasm_v128_load(gv + j)));
    }
  }
  for (int L = 0; L < layers; L++) {
    const float *wqkv = p, *bqkv = wqkv + dm * d3, *wo = bqkv + d3, *bo = wo + dm * dm, *w1 = bo + dm, *b1 = w1 + dm * ff;
    const float *w2 = b1 + ff, *b2 = w2 + ff * dm;
    p = b2 + dm;
    norm(h, R, dm, a);
    matmul(a, R, dm, wqkv, d3, bqkv, qkv);
    for (int b = 0; b < B; b++) {
      const float *base = qkv + b * n * d3;
      for (int hd = 0; hd < heads; hd++) {
        for (int qi = 0; qi < n; qi++) {
          const float *qv = base + qi * d3 + hd * dh;
          float mx = -3.0e38f;
          for (int kj = 0; kj < n; kj++) {
            const float *kv = base + kj * d3 + dm + hd * dh;
            v128_t s = wasm_f32x4_splat(0.0f);
            for (int e = 0; e < dh; e += 4) s = madd(s, wasm_v128_load(qv + e), wasm_v128_load(kv + e));
            float sc = hsum(s);
            att[kj] = sc;
            if (sc > mx) mx = sc;
          }
          float sum = 0.0f;
          for (int kj = 0; kj < n; kj++) { att[kj] = exp_neg(att[kj] - mx); sum += att[kj]; }
          float inv = 1.0f / sum;
          float *orow = o + (b * n + qi) * dm + hd * dh;
          for (int e = 0; e < dh; e += 4) wasm_v128_store(orow + e, wasm_f32x4_splat(0.0f));
          for (int kj = 0; kj < n; kj++) {
            v128_t w = wasm_f32x4_splat(att[kj] * inv);
            const float *vv = base + kj * d3 + 2 * dm + hd * dh;
            for (int e = 0; e < dh; e += 4) wasm_v128_store(orow + e, madd(wasm_v128_load(orow + e), w, wasm_v128_load(vv + e)));
          }
        }
      }
    }
    matmul(o, R, dm, wo, dm, bo, t);
    for (int r = 0; r < R * dm; r += 4) wasm_v128_store(h + r, wasm_f32x4_add(wasm_v128_load(h + r), wasm_v128_load(t + r)));
    norm(h, R, dm, a);
    matmul(a, R, dm, w1, ff, b1, u);
    for (int r = 0; r < R * ff; r += 4) wasm_v128_store(u + r, wasm_f32x4_max(wasm_v128_load(u + r), wasm_f32x4_splat(0.0f)));
    matmul(u, R, ff, w2, dm, b2, t);
    for (int r = 0; r < R * dm; r += 4) wasm_v128_store(h + r, wasm_f32x4_add(wasm_v128_load(h + r), wasm_v128_load(t + r)));
  }
  const float *wPool = p, *bG = wPool + dm * dm, *wGGlob = bG + dm, *v1w = wGGlob + NG * dm, *v1b = v1w + dm * dm;
  const float *v2w = v1b + dm, *v2b = v2w + dm;
  norm(h, R, dm, a);
  float *g = t, *v1 = t + dm;
  for (int b = 0; b < B; b++) {
    for (int j = 0; j < dm; j += 4) {
      v128_t s = wasm_f32x4_splat(0.0f);
      for (int q = 0; q < n; q++) s = wasm_f32x4_add(s, wasm_v128_load(a + (b * n + q) * dm + j));
      wasm_v128_store(pooled + j, wasm_f32x4_div(s, wasm_f32x4_splat((float)n)));
    }
    matmul(pooled, 1, dm, wPool, dm, bG, g);
    matmul(glob + b * NG, 1, NG, wGGlob, dm, 0, g2);
    for (int j = 0; j < dm; j += 4)
      wasm_v128_store(g + j, wasm_f32x4_max(wasm_f32x4_add(wasm_v128_load(g + j), wasm_v128_load(g2 + j)), wasm_f32x4_splat(0.0f)));
    matmul(g, 1, dm, v1w, dm, v1b, v1);
    v128_t s = wasm_f32x4_splat(0.0f);
    for (int j = 0; j < dm; j += 4) s = madd(s, wasm_f32x4_max(wasm_v128_load(v1 + j), wasm_f32x4_splat(0.0f)), wasm_v128_load(v2w + j));
    out[b] = hsum(s) + v2b[0];
  }
}

/* Floats of scratch value_batch needs for B positions of n planets. */
__attribute__((export_name("scratch_size")))
int scratch_size(int dm, int ff, int B, int n) {
  int R = B * n;
  return R * dm * 4 + R * 3 * dm + R * ff + ((n + 3) & ~3) + 3 * dm;
}
