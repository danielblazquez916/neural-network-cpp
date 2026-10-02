#pragma once

#include <iostream>
#include <vector>
#include <random>

class Neuron
{
public:
	std::vector<float> weights;
	std::vector<float> inputs;

	std::vector<float> accumulated_gradient_weights;
	float accumulated_gradient_bias;

	float bias;
	float output_pre_ReLU;
	bool first_time;

	Neuron();

	void init(size_t inputs_size, std::normal_distribution<float>& dist);

	void batching(float logit_gradient, int iter, int batch_size, int images_size, float learning_rate);

	void update_parameters(int batch_size, float learning_rate);

	float get_output(std::vector<float>& inputs, std::normal_distribution<float>& dist, bool ReLU);
};