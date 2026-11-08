#pragma once

#include <cstdint>
#include <vector>

struct AugmentConfig
{
	bool enabled = false;
	float max_shift = 2.f;
	float max_rotation = 10.f;
	float max_scale = 0.10f;
	float max_shear = 0.10f;
	float elastic_alpha = 10.f;
	float elastic_sigma = 4.f;

	bool operator==(const AugmentConfig& other) const
	{
		return enabled == other.enabled && max_shift == other.max_shift && max_rotation == other.max_rotation &&
			max_scale == other.max_scale && max_shear == other.max_shear &&
			elastic_alpha == other.elastic_alpha && elastic_sigma == other.elastic_sigma;
	}

	bool operator!=(const AugmentConfig& other) const { return !(*this == other); }
};

class Augmenter
{
public:
	static constexpr int size = 28;
	static constexpr int pool_size = 256;

	Augmenter();

	Augmenter(const AugmentConfig& config, uint32_t seed);

	const AugmentConfig& config() const { return config_; }

	void apply(const float* src, float* dst, uint64_t seed) const;

private:
	AugmentConfig config_;
	std::vector<float> fields_;
};
