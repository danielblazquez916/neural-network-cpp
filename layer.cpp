#include "layer.hpp"

Layer::Layer(size_t layer_size, std::vector<Neuron>* neurons, bool ReLU) : ReLU(ReLU), first_time(true)
{
	if (neurons == nullptr) this->neurons.resize(layer_size);
	else this->neurons = std::move(*neurons);
}

void Layer::gradient(std::vector<float>& next_gradients, std::vector<Neuron>& next_neurons)
{
	gradients.clear();

	for (int i = 0; i < neurons.size(); ++i) // neuronas ACTUALES
	{
		float gradient = 0.f;

		for (int j = 0; j < next_neurons.size(); ++j) // neuronas SIGUIENTES
		{
			gradient += next_neurons[j].weights[i] * next_gradients[j];
		}

		gradient *= (neurons[i].output_pre_ReLU > 0.f);

		// neurons[i].update_parameters(gradient);
		gradients.push_back(gradient);
	}
}

void Layer::last_layer_gradient(std::vector<float>& probs, int label)
{
	gradients.clear();
	for (int i = 0; i < neurons.size(); ++i)
	{
		float one_hot = i == label;
		float logit_gradient = probs[i] - one_hot;

		// neurons[i].update_parameters(logit_gradient);
		gradients.push_back(logit_gradient);
	}
}

void Layer::start(std::vector<float> inputs)
{
	float stddev = std::sqrt(2.0f / inputs.size());
	std::normal_distribution<float> dist(0.0f, stddev);

	outputs.clear();
	for (size_t i = 0; i < neurons.size(); ++i)
	{
		outputs.push_back(neurons[i].get_output(inputs, dist, ReLU));
	}
}