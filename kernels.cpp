#include "kernels.hpp"
#include <atomic>

#if defined(NN_X86_64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace kernels
{
	namespace
	{
		std::atomic<bool> simd_allowed{ true };

		bool detect_avx2()
		{
#if defined(NN_X86_64) && (defined(__GNUC__) || defined(__clang__))
			__builtin_cpu_init();

			return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(NN_X86_64) && defined(_MSC_VER)
			int info[4] = {};

			__cpuid(info, 0);

			if (info[0] < 7) return false;

			__cpuid(info, 1);

			const bool fma = ((info[2] >> 12) & 1) != 0;
			const bool osxsave = ((info[2] >> 27) & 1) != 0;
			const bool avx = ((info[2] >> 28) & 1) != 0;

			if (!(fma && osxsave && avx)) return false;

			if ((_xgetbv(0) & 6) != 6) return false;

			__cpuidex(info, 7, 0);

			return ((info[1] >> 5) & 1) != 0;
#else
			return false;
#endif
		}

		bool use_avx2()
		{
			return simd_allowed.load(std::memory_order_relaxed) && cpu_has_avx2();
		}
	}

	bool cpu_has_avx2()
	{
		static const bool supported = detect_avx2();

		return supported;
	}

	void set_simd_enabled(bool enabled)
	{
		simd_allowed.store(enabled, std::memory_order_relaxed);
	}

	bool simd_enabled()
	{
		return use_avx2();
	}

	void linear_forward(const float* X, const float* W, const float* bias, float* Y, int batch, int in, int out)
	{
#if defined(NN_X86_64)
		if (use_avx2())
		{
			avx2::linear_forward(X, W, bias, Y, batch, in, out);
			return;
		}
#endif
		scalar::linear_forward(X, W, bias, Y, batch, in, out);
	}

	void linear_backward_input(const float* dY, const float* W, float* dX, int batch, int in, int out)
	{
#if defined(NN_X86_64)
		if (use_avx2())
		{
			avx2::linear_backward_input(dY, W, dX, batch, in, out);
			return;
		}
#endif
		scalar::linear_backward_input(dY, W, dX, batch, in, out);
	}

	void linear_backward_weights(const float* X, const float* dY, float* dW, float* db, int batch, int in, int out)
	{
#if defined(NN_X86_64)
		if (use_avx2())
		{
			avx2::linear_backward_weights(X, dY, dW, db, batch, in, out);
			return;
		}
#endif
		scalar::linear_backward_weights(X, dY, dW, db, batch, in, out);
	}

	void adamw_update(float* w, const float* g, float* m, float* v, size_t n,
		float beta1, float beta2, float epsilon, float step_size, float inv_sqrt_bias_correction2, float shrink)
	{
#if defined(NN_X86_64)
		if (use_avx2())
		{
			avx2::adamw_update(w, g, m, v, n, beta1, beta2, epsilon, step_size, inv_sqrt_bias_correction2, shrink);
			return;
		}
#endif
		scalar::adamw_update(w, g, m, v, n, beta1, beta2, epsilon, step_size, inv_sqrt_bias_correction2, shrink);
	}
}
