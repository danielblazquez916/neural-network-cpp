#include "utils.hpp"
#include <cmath>
#include <limits>

std::pair<float, int> get_biggest(const std::vector<float>& nums)
{
	float biggest = std::numeric_limits<float>::lowest();
	int index = 0;

	for (int i = 0; i < static_cast<int>(nums.size()); ++i)
	{
		if (nums[i] > biggest)
		{
			biggest = nums[i];
			index = i;
		}
	}
	return { biggest, index };
}

uint32_t read_uint32(std::ifstream& file)
{
	uint8_t bytes[4] = {};
	file.read(reinterpret_cast<char*>(bytes), 4);

	return (static_cast<uint32_t>(bytes[0]) << 24) |
		(static_cast<uint32_t>(bytes[1]) << 16) |
		(static_cast<uint32_t>(bytes[2]) << 8) |
		static_cast<uint32_t>(bytes[3]);
}

double softmax_cross_entropy(const float* logits, int batch, int classes, const uint8_t* labels,
	float* probs, float* grad, int* predictions, float label_smoothing)
{
	double total = 0.0;
	const float inv_batch = 1.f / static_cast<float>(batch);
	const float smooth_target = label_smoothing / static_cast<float>(classes);
	const float true_target = 1.f - label_smoothing + smooth_target;

	for (int s = 0; s < batch; ++s)
	{
		const float* z = logits + static_cast<size_t>(s) * classes;
		const int label = labels[s];

		if (label >= classes) return std::numeric_limits<double>::quiet_NaN();

		int best = 0;
		float biggest = z[0];

		for (int c = 1; c < classes; ++c)
		{
			if (z[c] > biggest)
			{
				biggest = z[c];
				best = c;
			}
		}

		double sum = 0.0;

		for (int c = 0; c < classes; ++c) sum += std::exp(static_cast<double>(z[c]) - biggest);

		const double log_sum = static_cast<double>(biggest) + std::log(sum);

		total += log_sum - static_cast<double>(z[label]);

		for (int c = 0; c < classes; ++c)
		{
			const float p = static_cast<float>(std::exp(static_cast<double>(z[c]) - log_sum));
			const size_t index = static_cast<size_t>(s) * classes + c;

			if (probs) probs[index] = p;
			if (grad) grad[index] = (p - (c == label ? true_target : smooth_target)) * inv_batch;
		}

		if (predictions) predictions[s] = best;
	}

	return total;
}

uint32_t random_below(std::mt19937& rng, uint32_t n)
{
	if (n <= 1) return 0;

	const uint32_t limit = 0xFFFFFFFFu - (0xFFFFFFFFu % n);
	uint32_t value = static_cast<uint32_t>(rng());

	while (value >= limit) value = static_cast<uint32_t>(rng());

	return value % n;
}

float random_uniform(std::mt19937& rng)
{
	return static_cast<float>(static_cast<uint32_t>(rng()) >> 8) * (1.f / 16777216.f);
}

float random_normal(std::mt19937& rng)
{
	const float u1 = 1.f - random_uniform(rng);
	const float u2 = random_uniform(rng);

	return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530717958647692f * u2);
}

void shuffle_indices(std::vector<size_t>& indices, std::mt19937& rng)
{
	for (size_t i = indices.size(); i > 1; --i)
	{
		const size_t j = random_below(rng, static_cast<uint32_t>(i));

		std::swap(indices[i - 1], indices[j]);
	}
}
