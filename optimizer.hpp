#pragma once

#include <cstddef>
#include <vector>

enum class OptimizerType
{
	SGD,
	AdamW
};

struct OptimizerConfig
{
	OptimizerType type = OptimizerType::AdamW;
	float weight_decay = 1e-2f;
	float momentum = 0.9f;
	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float epsilon = 1e-8f;
};

struct Param
{
	std::vector<float> w;
	std::vector<float> g;
	std::vector<float> m;
	std::vector<float> v;
	bool decay = false;

	void init(size_t size, bool apply_decay, float value = 0.f);
	void zero_grad();
	void reset_state();

	void step(const OptimizerConfig& config, float learning_rate, long step);
};
