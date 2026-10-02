#pragma once

#include "layer.hpp"
#include <array>
#include <algorithm>
#include <random>

class neural_network
{
public:
	std::vector<Layer> layers;
	std::vector<std::pair<std::vector<float>, uint8_t>> images;
	int iter;
	float average_loss;
	int corrects;
	int batch_size;
	float learning_rate;

	neural_network(int batch_size, float learning_rate);
	~neural_network() = default;

	void shuffle();

	void process(bool prob_flag, bool bp_flag);

	void backprop(bool flag, std::vector<float>& probs);

	void probability(bool flag, std::vector<float>& probs);

	void serialize(const char* file_name);

	void deserialize(const char* file_name);

	void read_images(const std::string& type);

	void restart_values();

	std::vector<float> apply_stable_softmax(std::vector<float>& logits);
};