#pragma once

#include "optimizer.hpp"
#include <cstdint>
#include <random>
#include <vector>

class Layer
{
public:
	int in_features;
	int out_features;
	bool ReLU;
	bool batchnorm;
	float dropout;

	Param weights;
	Param bias;
	Param gamma;
	Param beta;
	std::vector<float> running_mean;
	std::vector<float> running_var;

	std::vector<float> outputs;
	std::vector<float> gradients;

	Layer();
	Layer(int in_features, int out_features, bool ReLU = true, bool batchnorm = false, float dropout = 0.f);

	void init_parameters(std::mt19937& rng);

	const std::vector<float>& forward(const float* inputs, int batch, bool training, std::mt19937& rng);

	const float* backward(const float* grad_outputs, bool need_input_gradient);

	void zero_gradients();
	void update(const OptimizerConfig& config, float learning_rate, long step);
	void reset_optimizer_state();
	bool is_finite() const;

private:
	static constexpr float bn_epsilon = 1e-5f;
	static constexpr float bn_momentum = 0.1f;

	const float* inputs_;
	int batch_;
	bool dropout_active_;

	std::vector<float> normalized_;
	std::vector<float> inv_std_;
	std::vector<float> dropout_mask_;
	std::vector<float> scratch_;
};
