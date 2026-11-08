#pragma once

#include <cmath>
#include <cstddef>

#if defined(_MSC_VER)
#define NN_RESTRICT __restrict
#else
#define NN_RESTRICT __restrict__
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define NN_X86_64 1
#endif

namespace kernels
{
	namespace scalar
	{
		inline float dot(const float* NN_RESTRICT a, const float* NN_RESTRICT b, int n)
		{
			float acc[8] = {};
			int i = 0;

			for (; i + 8 <= n; i += 8)
			{
				for (int k = 0; k < 8; ++k) acc[k] += a[i + k] * b[i + k];
			}

			float sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));

			for (; i < n; ++i) sum += a[i] * b[i];

			return sum;
		}

		inline void axpy(float* NN_RESTRICT y, const float* NN_RESTRICT x, float alpha, int n)
		{
			for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
		}

		inline void linear_forward(const float* X, const float* W, const float* bias, float* Y, int batch, int in, int out)
		{
			for (int o = 0; o < out; ++o)
			{
				const float* w = W + static_cast<size_t>(o) * in;
				const float b = bias ? bias[o] : 0.f;

				for (int s = 0; s < batch; ++s)
				{
					Y[static_cast<size_t>(s) * out + o] = b + dot(X + static_cast<size_t>(s) * in, w, in);
				}
			}
		}

		inline void linear_backward_input(const float* dY, const float* W, float* dX, int batch, int in, int out)
		{
			for (int s = 0; s < batch; ++s)
			{
				float* NN_RESTRICT dx = dX + static_cast<size_t>(s) * in;
				const float* dy = dY + static_cast<size_t>(s) * out;

				for (int i = 0; i < in; ++i) dx[i] = 0.f;

				int o = 0;

				for (; o + 4 <= out; o += 4)
				{
					const float a0 = dy[o], a1 = dy[o + 1], a2 = dy[o + 2], a3 = dy[o + 3];
					const float* NN_RESTRICT w0 = W + static_cast<size_t>(o) * in;
					const float* NN_RESTRICT w1 = w0 + in;
					const float* NN_RESTRICT w2 = w1 + in;
					const float* NN_RESTRICT w3 = w2 + in;

					for (int i = 0; i < in; ++i) dx[i] += a0 * w0[i] + a1 * w1[i] + a2 * w2[i] + a3 * w3[i];
				}

				for (; o < out; ++o) axpy(dx, W + static_cast<size_t>(o) * in, dy[o], in);
			}
		}

		inline void linear_backward_weights(const float* X, const float* dY, float* dW, float* db, int batch, int in, int out)
		{
			for (int o = 0; o < out; ++o)
			{
				float* NN_RESTRICT dw = dW + static_cast<size_t>(o) * in;
				int s = 0;

				for (; s + 4 <= batch; s += 4)
				{
					const float a0 = dY[static_cast<size_t>(s) * out + o];
					const float a1 = dY[static_cast<size_t>(s + 1) * out + o];
					const float a2 = dY[static_cast<size_t>(s + 2) * out + o];
					const float a3 = dY[static_cast<size_t>(s + 3) * out + o];
					const float* NN_RESTRICT x0 = X + static_cast<size_t>(s) * in;
					const float* NN_RESTRICT x1 = x0 + in;
					const float* NN_RESTRICT x2 = x1 + in;
					const float* NN_RESTRICT x3 = x2 + in;

					for (int i = 0; i < in; ++i) dw[i] += a0 * x0[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i];
				}

				for (; s < batch; ++s) axpy(dw, X + static_cast<size_t>(s) * in, dY[static_cast<size_t>(s) * out + o], in);

				if (db)
				{
					float sum = 0.f;

					for (int k = 0; k < batch; ++k) sum += dY[static_cast<size_t>(k) * out + o];

					db[o] += sum;
				}
			}
		}

		inline void adamw_update(float* NN_RESTRICT w, const float* NN_RESTRICT g, float* NN_RESTRICT m, float* NN_RESTRICT v, size_t n,
			float beta1, float beta2, float epsilon, float step_size, float inv_sqrt_bias_correction2, float shrink)
		{
			for (size_t i = 0; i < n; ++i)
			{
				const float grad = g[i];

				m[i] = beta1 * m[i] + (1.f - beta1) * grad;
				v[i] = beta2 * v[i] + (1.f - beta2) * grad * grad;

				const float denominator = std::sqrt(v[i]) * inv_sqrt_bias_correction2 + epsilon;

				w[i] = w[i] * shrink - step_size * m[i] / denominator;
			}
		}
	}

#if defined(NN_X86_64)
	namespace avx2
	{
		void linear_forward(const float* X, const float* W, const float* bias, float* Y, int batch, int in, int out);
		void linear_backward_input(const float* dY, const float* W, float* dX, int batch, int in, int out);
		void linear_backward_weights(const float* X, const float* dY, float* dW, float* db, int batch, int in, int out);
		void adamw_update(float* w, const float* g, float* m, float* v, size_t n,
			float beta1, float beta2, float epsilon, float step_size, float inv_sqrt_bias_correction2, float shrink);
	}
#endif

	bool cpu_has_avx2();

	void set_simd_enabled(bool enabled);
	bool simd_enabled();

	void linear_forward(const float* X, const float* W, const float* bias, float* Y, int batch, int in, int out);
	void linear_backward_input(const float* dY, const float* W, float* dX, int batch, int in, int out);
	void linear_backward_weights(const float* X, const float* dY, float* dW, float* db, int batch, int in, int out);
	void adamw_update(float* w, const float* g, float* m, float* v, size_t n,
		float beta1, float beta2, float epsilon, float step_size, float inv_sqrt_bias_correction2, float shrink);
}
