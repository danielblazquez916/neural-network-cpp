#include "neuron.hpp"
#include <iostream>

Neuron::Neuron() : bias(0.f), output_pre_ReLU(0.f), first_time(true), accumulated_gradient_bias(0.f) {}

void Neuron::init(size_t inputs_size, std::normal_distribution<float>& dist)
{
	static std::mt19937 gen(std::random_device{}());

	for (int i = 0; i < inputs_size; ++i)
	{
		weights.push_back(dist(gen));
	}
}

void Neuron::batching(float logit_gradient, int iter, int batch_size, int images_size, float learning_rate)
{
	for (int i = 0; i < inputs.size(); ++i)
	{
		float weight_gradient = logit_gradient * inputs[i];

		if (first_time) accumulated_gradient_weights.push_back(weight_gradient);
		else accumulated_gradient_weights[i] += weight_gradient;
	}

	accumulated_gradient_bias += logit_gradient;
	first_time = false;

	if ((iter + 1) % batch_size != 0 && iter != images_size - 1) return;

	int current_batch_size =
		(iter == images_size - 1 && images_size % batch_size != 0)
		? images_size % batch_size
		: batch_size;
	
	update_parameters(current_batch_size, learning_rate);
	first_time = true;

	accumulated_gradient_weights.clear();
	accumulated_gradient_bias = 0.f;
}

void Neuron::update_parameters(int batch_size, float learning_rate)
{
	for (int i = 0; i < weights.size(); ++i)
	{
		// float weight_gradient = logit_gradient * inputs[i];
		weights[i] -= (accumulated_gradient_weights[i]/static_cast<float>(batch_size)) * learning_rate;
	}

	bias -= (accumulated_gradient_bias / static_cast<float>(batch_size)) * learning_rate;
}

float Neuron::get_output(std::vector<float>& inputs, std::normal_distribution<float>& dist, bool ReLU)
{
	if (weights.empty()) init(inputs.size(), dist);
	this->inputs = inputs;

	float output = 0.f;
	for (int i = 0; i < inputs.size(); ++i)
	{
		output += weights[i] * inputs[i];
	}
	output += bias;
	output_pre_ReLU = output;
	if (ReLU) output = output >= 0 ? output : 0; // ReLU

	return output;
}

