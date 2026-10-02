#pragma once

#include "neuron.hpp"

class Layer
{
public:
	std::vector<Neuron> neurons;
	bool ReLU;
	std::vector<float> gradients;
	std::vector<float> outputs;
	bool first_time;

	Layer(size_t layer_size, std::vector<Neuron>* neurons = nullptr, bool ReLU = true);

	void gradient(std::vector<float>& next_gradients, std::vector<Neuron>& next_neurons);

	void last_layer_gradient(std::vector<float>& probs, int label);

	void start(std::vector<float> inputs);
};