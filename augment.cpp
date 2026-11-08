#include "augment.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>

namespace
{
	constexpr int pixels = Augmenter::size * Augmenter::size;
	constexpr int padded_size = Augmenter::size + 2;
	constexpr float center = (Augmenter::size - 1) / 2.f;

	struct Stream
	{
		uint64_t state;

		uint64_t next() { return mix64(state++); }

		float uniform() { return static_cast<float>(next() >> 40) * (1.f / 16777216.f); }

		float symmetric() { return 2.f * uniform() - 1.f; }
	};

	void gaussian_blur(std::vector<float>& field, float sigma)
	{
		const int size = Augmenter::size;
		const int radius = std::max(1, static_cast<int>(std::ceil(3.f * sigma)));
		std::vector<float> kernel(2 * radius + 1);
		float sum = 0.f;

		for (int i = -radius; i <= radius; ++i)
		{
			kernel[i + radius] = std::exp(-0.5f * static_cast<float>(i * i) / (sigma * sigma));
			sum += kernel[i + radius];
		}

		for (float& k : kernel) k /= sum;

		std::vector<float> temp(field.size(), 0.f);

		for (int y = 0; y < size; ++y)
		{
			for (int x = 0; x < size; ++x)
			{
				float acc = 0.f;

				for (int i = -radius; i <= radius; ++i)
				{
					const int xx = x + i;

					if (xx >= 0 && xx < size) acc += kernel[i + radius] * field[y * size + xx];
				}

				temp[y * size + x] = acc;
			}
		}

		for (int y = 0; y < size; ++y)
		{
			for (int x = 0; x < size; ++x)
			{
				float acc = 0.f;

				for (int i = -radius; i <= radius; ++i)
				{
					const int yy = y + i;

					if (yy >= 0 && yy < size) acc += kernel[i + radius] * temp[yy * size + x];
				}

				field[y * size + x] = acc;
			}
		}
	}
}

Augmenter::Augmenter() = default;

Augmenter::Augmenter(const AugmentConfig& config, uint32_t seed) : config_(config)
{
	if (!(config_.enabled && config_.elastic_alpha > 0.f && config_.elastic_sigma > 0.f)) return;

	std::mt19937 rng(seed);

	fields_.assign(static_cast<size_t>(pool_size) * 2 * pixels, 0.f);

	for (int f = 0; f < pool_size * 2; ++f)
	{
		std::vector<float> field(pixels);

		for (float& value : field) value = 2.f * random_uniform(rng) - 1.f;

		gaussian_blur(field, config_.elastic_sigma);

		for (int i = 0; i < pixels; ++i) fields_[static_cast<size_t>(f) * pixels + i] = field[i] * config_.elastic_alpha;
	}
}

void Augmenter::apply(const float* src, float* dst, uint64_t seed) const
{
	const int size = Augmenter::size;
	Stream random{ seed };

	const float angle = random.symmetric() * config_.max_rotation * 0.017453292519943295f;
	const float scale_x = 1.f + random.symmetric() * config_.max_scale;
	const float scale_y = 1.f + random.symmetric() * config_.max_scale;
	const float shear = random.symmetric() * config_.max_shear;
	const float shift_x = random.symmetric() * config_.max_shift;
	const float shift_y = random.symmetric() * config_.max_shift;

	const float c = std::cos(angle);
	const float s = std::sin(angle);
	const float m00 = c * scale_x;
	const float m01 = c * shear - s * scale_y;
	const float m10 = s * scale_x;
	const float m11 = s * shear + c * scale_y;
	const float inv_det = 1.f / (m00 * m11 - m01 * m10);

	const float i00 = m11 * inv_det;
	const float i01 = -m01 * inv_det;
	const float i10 = -m10 * inv_det;
	const float i11 = m00 * inv_det;

	const float* field_x = nullptr;
	const float* field_y = nullptr;
	float sign_x = 1.f;
	float sign_y = 1.f;

	if (!fields_.empty())
	{
		const uint64_t bits = random.next();
		const size_t pick = static_cast<size_t>(bits % static_cast<uint64_t>(pool_size));

		field_x = fields_.data() + (pick * 2) * pixels;
		field_y = field_x + pixels;
		sign_x = (bits >> 32) & 1 ? 1.f : -1.f;
		sign_y = (bits >> 33) & 1 ? 1.f : -1.f;
	}

	float padded[padded_size * padded_size] = {};

	for (int y = 0; y < size; ++y)
	{
		std::copy(src + y * size, src + (y + 1) * size, padded + (y + 1) * padded_size + 1);
	}

	const float limit = static_cast<float>(size);

	for (int y = 0; y < size; ++y)
	{
		for (int x = 0; x < size; ++x)
		{
			const float u = static_cast<float>(x) - center - shift_x;
			const float v = static_cast<float>(y) - center - shift_y;
			float px = i00 * u + i01 * v + center;
			float py = i10 * u + i11 * v + center;

			if (field_x)
			{
				px += sign_x * field_x[y * size + x];
				py += sign_y * field_y[y * size + x];
			}

			px = std::min(std::max(px, -1.f), limit - 1e-3f);
			py = std::min(std::max(py, -1.f), limit - 1e-3f);

			const float fx = std::floor(px);
			const float fy = std::floor(py);
			const float tx = px - fx;
			const float ty = py - fy;
			const float* p = padded + (static_cast<int>(fy) + 1) * padded_size + static_cast<int>(fx) + 1;

			const float top = p[0] + tx * (p[1] - p[0]);
			const float bottom = p[padded_size] + tx * (p[padded_size + 1] - p[padded_size]);

			dst[y * size + x] = top + ty * (bottom - top);
		}
	}
}
