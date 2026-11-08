#pragma once

#include "augment.hpp"
#include "layer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct EpochStats
{
	static constexpr int classes = 10;

	double loss = 0.0;
	double accuracy = 0.0;
	size_t samples = 0;
	bool diverged = false;
	std::array<std::array<int, classes>, classes> confusion = {};

	double accuracy_error() const;
};

using Sample = std::pair<std::vector<float>, uint8_t>;
using ImageSet = std::vector<Sample>;

class neural_network
{
public:
	static constexpr int input_size = 28 * 28;
	static constexpr int num_classes = 10;

	std::vector<Layer> layers;
	ImageSet images;
	ImageSet validation;

	int batch_size;
	float learning_rate;
	float lr_decay;
	int schedule_epochs;
	int warmup_steps;
	float label_smoothing;
	OptimizerConfig optimizer;
	AugmentConfig augment;

	int epochs_trained;
	long step_count;
	size_t holdout_every;
	float input_mean;
	float input_std;
	std::mt19937 rng;

	neural_network(int batch_size, float learning_rate);
	~neural_network() = default;

	EpochStats process(bool prob_flag, bool bp_flag);

	EpochStats evaluate(const ImageSet& data);

	void hold_out_validation(size_t every = 12);

	void begin_training_session(int epochs);

	const std::vector<float>& forward(const float* inputs, int batch, bool training);

	void backward(const float* dlogits);

	void apply_gradients();

	float current_learning_rate() const;

	void build_default_layers();

	bool serialize(const char* file_name) const;

	bool deserialize(const char* file_name, bool create_if_missing = true);

	bool read_images(const std::string& type);

private:
	EpochStats run_pass(const ImageSet& data, bool training, bool prob_flag, bool report);

	Augmenter augmenter_;
	int session_start_epoch_ = 0;
	float epoch_progress_ = 0.f;
};
