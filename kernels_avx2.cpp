#include "kernels.hpp"

#if defined(NN_X86_64)

#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define NN_AVX2 __attribute__((target("avx2,fma")))
#else
#define NN_AVX2
#endif

namespace kernels
{
	namespace avx2
	{
		namespace
		{
			constexpr double parallel_threshold = 1.5e6;

			NN_AVX2 inline float hsum(__m256 v)
			{
				__m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
				lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
				lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));

				return _mm_cvtss_f32(lo);
			}

			NN_AVX2 inline float dot(const float* a, const float* b, int n)
			{
				__m256 s0 = _mm256_setzero_ps();
				__m256 s1 = _mm256_setzero_ps();
				int i = 0;

				for (; i + 16 <= n; i += 16)
				{
					s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
					s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
				}

				for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);

				float sum = hsum(_mm256_add_ps(s0, s1));

				for (; i < n; ++i) sum += a[i] * b[i];

				return sum;
			}

			NN_AVX2 inline void forward_tile_4x3(const float* x, size_t ldx, const float* w, size_t ldw, int K, float* y, size_t ldy, const float* bias)
			{
				const float* x0 = x;
				const float* x1 = x + ldx;
				const float* x2 = x + 2 * ldx;
				const float* x3 = x + 3 * ldx;
				const float* w0 = w;
				const float* w1 = w + ldw;
				const float* w2 = w + 2 * ldw;

				__m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
				__m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
				__m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
				__m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

				int k = 0;

				for (; k + 8 <= K; k += 8)
				{
					const __m256 a0 = _mm256_loadu_ps(x0 + k);
					const __m256 a1 = _mm256_loadu_ps(x1 + k);
					const __m256 a2 = _mm256_loadu_ps(x2 + k);
					const __m256 a3 = _mm256_loadu_ps(x3 + k);

					__m256 b = _mm256_loadu_ps(w0 + k);
					c00 = _mm256_fmadd_ps(a0, b, c00);
					c10 = _mm256_fmadd_ps(a1, b, c10);
					c20 = _mm256_fmadd_ps(a2, b, c20);
					c30 = _mm256_fmadd_ps(a3, b, c30);

					b = _mm256_loadu_ps(w1 + k);
					c01 = _mm256_fmadd_ps(a0, b, c01);
					c11 = _mm256_fmadd_ps(a1, b, c11);
					c21 = _mm256_fmadd_ps(a2, b, c21);
					c31 = _mm256_fmadd_ps(a3, b, c31);

					b = _mm256_loadu_ps(w2 + k);
					c02 = _mm256_fmadd_ps(a0, b, c02);
					c12 = _mm256_fmadd_ps(a1, b, c12);
					c22 = _mm256_fmadd_ps(a2, b, c22);
					c32 = _mm256_fmadd_ps(a3, b, c32);
				}

				float r[4][3] =
				{
					{ hsum(c00), hsum(c01), hsum(c02) },
					{ hsum(c10), hsum(c11), hsum(c12) },
					{ hsum(c20), hsum(c21), hsum(c22) },
					{ hsum(c30), hsum(c31), hsum(c32) }
				};

				for (; k < K; ++k)
				{
					const float xs[4] = { x0[k], x1[k], x2[k], x3[k] };
					const float ws[3] = { w0[k], w1[k], w2[k] };

					for (int i = 0; i < 4; ++i)
					{
						for (int j = 0; j < 3; ++j) r[i][j] += xs[i] * ws[j];
					}
				}

				for (int i = 0; i < 4; ++i)
				{
					for (int j = 0; j < 3; ++j) y[static_cast<size_t>(i) * ldy + j] = (bias ? bias[j] : 0.f) + r[i][j];
				}
			}

			NN_AVX2 inline void broadcast_tile_4x16(const float* A, size_t sar, size_t sak, const float* B, size_t ldb, int K, float* C, size_t ldc, bool accumulate)
			{
				const float* a0 = A;
				const float* a1 = A + sar;
				const float* a2 = A + 2 * sar;
				const float* a3 = A + 3 * sar;

				__m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
				__m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
				__m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
				__m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

				for (int k = 0; k < K; ++k)
				{
					const float* bp = B + static_cast<size_t>(k) * ldb;
					const size_t offset = static_cast<size_t>(k) * sak;
					const __m256 b0 = _mm256_loadu_ps(bp);
					const __m256 b1 = _mm256_loadu_ps(bp + 8);

					__m256 a = _mm256_broadcast_ss(a0 + offset);
					c00 = _mm256_fmadd_ps(a, b0, c00);
					c01 = _mm256_fmadd_ps(a, b1, c01);

					a = _mm256_broadcast_ss(a1 + offset);
					c10 = _mm256_fmadd_ps(a, b0, c10);
					c11 = _mm256_fmadd_ps(a, b1, c11);

					a = _mm256_broadcast_ss(a2 + offset);
					c20 = _mm256_fmadd_ps(a, b0, c20);
					c21 = _mm256_fmadd_ps(a, b1, c21);

					a = _mm256_broadcast_ss(a3 + offset);
					c30 = _mm256_fmadd_ps(a, b0, c30);
					c31 = _mm256_fmadd_ps(a, b1, c31);
				}

				float* row0 = C;
				float* row1 = C + ldc;
				float* row2 = C + 2 * ldc;
				float* row3 = C + 3 * ldc;

				if (accumulate)
				{
					c00 = _mm256_add_ps(c00, _mm256_loadu_ps(row0));
					c01 = _mm256_add_ps(c01, _mm256_loadu_ps(row0 + 8));
					c10 = _mm256_add_ps(c10, _mm256_loadu_ps(row1));
					c11 = _mm256_add_ps(c11, _mm256_loadu_ps(row1 + 8));
					c20 = _mm256_add_ps(c20, _mm256_loadu_ps(row2));
					c21 = _mm256_add_ps(c21, _mm256_loadu_ps(row2 + 8));
					c30 = _mm256_add_ps(c30, _mm256_loadu_ps(row3));
					c31 = _mm256_add_ps(c31, _mm256_loadu_ps(row3 + 8));
				}

				_mm256_storeu_ps(row0, c00);
				_mm256_storeu_ps(row0 + 8, c01);
				_mm256_storeu_ps(row1, c10);
				_mm256_storeu_ps(row1 + 8, c11);
				_mm256_storeu_ps(row2, c20);
				_mm256_storeu_ps(row2 + 8, c21);
				_mm256_storeu_ps(row3, c30);
				_mm256_storeu_ps(row3 + 8, c31);
			}

			void weights_edge(const float* X, const float* dY, float* dW, int batch, int in, int out, int o0, int o1, int i0, int i1)
			{
				for (int o = o0; o < o1; ++o)
				{
					for (int i = i0; i < i1; ++i)
					{
						float sum = 0.f;

						for (int s = 0; s < batch; ++s) sum += dY[static_cast<size_t>(s) * out + o] * X[static_cast<size_t>(s) * in + i];

						dW[static_cast<size_t>(o) * in + i] += sum;
					}
				}
			}

			void input_edge(const float* dY, const float* W, float* dX, int in, int out, int s0, int s1, int i0, int i1)
			{
				for (int s = s0; s < s1; ++s)
				{
					for (int i = i0; i < i1; ++i)
					{
						float sum = 0.f;

						for (int o = 0; o < out; ++o) sum += dY[static_cast<size_t>(s) * out + o] * W[static_cast<size_t>(o) * in + i];

						dX[static_cast<size_t>(s) * in + i] = sum;
					}
				}
			}
		}

		NN_AVX2 void linear_forward(const float* X, const float* W, const float* bias, float* Y, int batch, int in, int out)
		{
			const size_t ldx = static_cast<size_t>(in);
			const size_t ldy = static_cast<size_t>(out);
			const int batch4 = batch & ~3;
			const int out3 = out - out % 3;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(static_cast<double>(batch) * in * out >= parallel_threshold)
#endif
			for (int o = 0; o < out3; o += 3)
			{
				const float* w = W + static_cast<size_t>(o) * in;
				const float* b = bias ? bias + o : nullptr;

				for (int s = 0; s < batch4; s += 4) forward_tile_4x3(X + static_cast<size_t>(s) * in, ldx, w, ldx, in, Y + static_cast<size_t>(s) * out + o, ldy, b);

				for (int s = batch4; s < batch; ++s)
				{
					for (int j = 0; j < 3; ++j) Y[static_cast<size_t>(s) * out + o + j] = (bias ? bias[o + j] : 0.f) + dot(X + static_cast<size_t>(s) * in, w + static_cast<size_t>(j) * in, in);
				}
			}

			for (int o = out3; o < out; ++o)
			{
				const float* w = W + static_cast<size_t>(o) * in;

				for (int s = 0; s < batch; ++s) Y[static_cast<size_t>(s) * out + o] = (bias ? bias[o] : 0.f) + dot(X + static_cast<size_t>(s) * in, w, in);
			}
		}

		NN_AVX2 void linear_backward_input(const float* dY, const float* W, float* dX, int batch, int in, int out)
		{
			const int in16 = in & ~15;
			const int batch4 = batch & ~3;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(static_cast<double>(batch) * in * out >= parallel_threshold)
#endif
			for (int i = 0; i < in16; i += 16)
			{
				for (int s = 0; s < batch4; s += 4)
				{
					broadcast_tile_4x16(dY + static_cast<size_t>(s) * out, static_cast<size_t>(out), 1, W + i, static_cast<size_t>(in), out,
						dX + static_cast<size_t>(s) * in + i, static_cast<size_t>(in), false);
				}
			}

			if (in16 < in) input_edge(dY, W, dX, in, out, 0, batch4, in16, in);
			if (batch4 < batch) input_edge(dY, W, dX, in, out, batch4, batch, 0, in);
		}

		NN_AVX2 void linear_backward_weights(const float* X, const float* dY, float* dW, float* db, int batch, int in, int out)
		{
			const int in16 = in & ~15;
			const int out4 = out & ~3;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(static_cast<double>(batch) * in * out >= parallel_threshold)
#endif
			for (int i = 0; i < in16; i += 16)
			{
				for (int o = 0; o < out4; o += 4)
				{
					broadcast_tile_4x16(dY + o, 1, static_cast<size_t>(out), X + i, static_cast<size_t>(in), batch,
						dW + static_cast<size_t>(o) * in + i, static_cast<size_t>(in), true);
				}
			}

			if (in16 < in) weights_edge(X, dY, dW, batch, in, out, 0, out4, in16, in);
			if (out4 < out) weights_edge(X, dY, dW, batch, in, out, out4, out, 0, in);

			if (db)
			{
				for (int s = 0; s < batch; ++s)
				{
					const float* row = dY + static_cast<size_t>(s) * out;

					for (int o = 0; o < out; ++o) db[o] += row[o];
				}
			}
		}

		NN_AVX2 void adamw_update(float* w, const float* g, float* m, float* v, size_t n,
			float beta1, float beta2, float epsilon, float step_size, float inv_sqrt_bias_correction2, float shrink)
		{
			const __m256 vb1 = _mm256_set1_ps(beta1);
			const __m256 vb1c = _mm256_set1_ps(1.f - beta1);
			const __m256 vb2 = _mm256_set1_ps(beta2);
			const __m256 vb2c = _mm256_set1_ps(1.f - beta2);
			const __m256 vinv = _mm256_set1_ps(inv_sqrt_bias_correction2);
			const __m256 veps = _mm256_set1_ps(epsilon);
			const __m256 vstep = _mm256_set1_ps(step_size);
			const __m256 vshrink = _mm256_set1_ps(shrink);

			const int block = 4096;
			const int blocks = static_cast<int>((n + block - 1) / block);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(n >= 100000)
#endif
			for (int b = 0; b < blocks; ++b)
			{
				const size_t begin = static_cast<size_t>(b) * block;
				const size_t end = begin + block < n ? begin + block : n;
				size_t i = begin;

				for (; i + 8 <= end; i += 8)
				{
					const __m256 grad = _mm256_loadu_ps(g + i);
					__m256 mm = _mm256_loadu_ps(m + i);
					__m256 vv = _mm256_loadu_ps(v + i);
					__m256 ww = _mm256_loadu_ps(w + i);

					mm = _mm256_fmadd_ps(vb1, mm, _mm256_mul_ps(vb1c, grad));
					vv = _mm256_fmadd_ps(vb2, vv, _mm256_mul_ps(vb2c, _mm256_mul_ps(grad, grad)));

					const __m256 denominator = _mm256_fmadd_ps(_mm256_sqrt_ps(vv), vinv, veps);

					ww = _mm256_sub_ps(_mm256_mul_ps(ww, vshrink), _mm256_div_ps(_mm256_mul_ps(vstep, mm), denominator));

					_mm256_storeu_ps(m + i, mm);
					_mm256_storeu_ps(v + i, vv);
					_mm256_storeu_ps(w + i, ww);
				}

				if (i < end) scalar::adamw_update(w + i, g + i, m + i, v + i, end - i, beta1, beta2, epsilon, step_size, inv_sqrt_bias_correction2, shrink);
			}
		}
	}
}

#endif
