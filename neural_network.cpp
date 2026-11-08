#include "neural_network.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace
{
	using json = nlohmann::json;

	constexpr size_t report_interval = 10000;
	constexpr int evaluation_batch = 256;
	constexpr uint32_t max_images = 10000000;

	double compact(float value)
	{
		char text[32];
		std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
		return std::strtod(text, nullptr);
	}

	json to_json(const std::vector<float>& values)
	{
		json array = json::array();

		for (float value : values) array.push_back(compact(value));

		return array;
	}

	std::vector<float> read_floats(const json& node, const char* key, size_t expected)
	{
		std::vector<float> values = node.at(key).get<std::vector<float>>();

		if (values.size() != expected)
		{
			throw std::runtime_error(std::string("'") + key + "' has " + std::to_string(values.size()) + " values, expected " + std::to_string(expected));
		}

		return values;
	}

	std::string format_probabilities(const float* probs, int prediction, int label)
	{
		std::ostringstream line;
		line << std::fixed << std::setprecision(2) << "[ ";

		for (int c = 0; c < neural_network::num_classes; ++c)
		{
			line << probs[c] * 100.f << "%";

			if (c + 1 < neural_network::num_classes) line << ", ";
		}

		line << " ] - Prediction: " << prediction << " - Real Number: " << label << '\n';

		return line.str();
	}
}

double EpochStats::accuracy_error() const
{
	if (samples < 2) return 0.0;

	return std::sqrt(accuracy * (1.0 - accuracy) / static_cast<double>(samples - 1));
}

neural_network::neural_network(int batch_size, float learning_rate)
	: batch_size(batch_size), learning_rate(learning_rate), lr_decay(0.85f), schedule_epochs(0), warmup_steps(100), label_smoothing(0.f),
	  optimizer(), augment(), epochs_trained(0), step_count(0), holdout_every(0), input_mean(0.1307f), input_std(0.3081f), rng(std::random_device{}())
{
	if (batch_size < 2) throw std::invalid_argument("neural_network: batch_size must be at least 2 (BatchNorm needs a batch)");
	if (!(learning_rate > 0.f)) throw std::invalid_argument("neural_network: learning_rate must be positive");
}

void neural_network::build_default_layers()
{
	layers.clear();

	layers.emplace_back(input_size, 512, true, true, 0.2f);
	layers.emplace_back(512, 256, true, true, 0.2f);
	layers.emplace_back(256, 128, true, true, 0.f);
	layers.emplace_back(128, num_classes, false, false, 0.f);

	for (Layer& layer : layers) layer.init_parameters(rng);

	step_count = 0;
}

const std::vector<float>& neural_network::forward(const float* inputs, int batch, bool training)
{
	const float* current = inputs;

	for (Layer& layer : layers) current = layer.forward(current, batch, training, rng).data();

	return layers.back().outputs;
}

void neural_network::backward(const float* dlogits)
{
	for (Layer& layer : layers) layer.zero_gradients();

	const float* gradient = dlogits;

	for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
	{
		gradient = layers[i].backward(gradient, i > 0);
	}
}

void neural_network::begin_training_session(int epochs)
{
	schedule_epochs = epochs > 0 ? epochs : 0;
	session_start_epoch_ = epochs_trained;
	epoch_progress_ = 0.f;
}

float neural_network::current_learning_rate() const
{
	if (schedule_epochs > 0)
	{
		const float done = static_cast<float>(epochs_trained - session_start_epoch_) + epoch_progress_;
		const float progress = std::min(1.f, std::max(0.f, done / static_cast<float>(schedule_epochs)));

		return learning_rate * 0.5f * (1.f + std::cos(3.14159265358979f * progress));
	}

	const float decayed = learning_rate * std::pow(lr_decay, static_cast<float>(epochs_trained));
	const float floor = learning_rate * 0.01f;

	return decayed < floor ? floor : decayed;
}

void neural_network::apply_gradients()
{
	++step_count;

	float lr = current_learning_rate();

	if (warmup_steps > 0 && step_count < warmup_steps) lr *= static_cast<float>(step_count) / static_cast<float>(warmup_steps);

	for (Layer& layer : layers) layer.update(optimizer, lr, step_count);
}

void neural_network::hold_out_validation(size_t every)
{
	if (every < 2) return;

	ImageSet kept;
	kept.reserve(images.size());

	for (size_t i = 0; i < images.size(); ++i)
	{
		(mix64(i) % every == 0 ? validation : kept).push_back(std::move(images[i]));
	}

	images = std::move(kept);
}

EpochStats neural_network::evaluate(const ImageSet& data)
{
	return run_pass(data, false, false, false);
}

EpochStats neural_network::process(bool prob_flag, bool bp_flag)
{
	if (layers.empty())
	{
		std::cout << "- Error: the network has no layers.\n";
		return EpochStats();
	}

	if (images.empty())
	{
		std::cout << "- There are no images to process.\n";
		return EpochStats();
	}

	const auto start_time = std::chrono::steady_clock::now();
	const EpochStats stats = run_pass(images, bp_flag, prob_flag, true);

	if (stats.diverged) return stats;

	if (stats.samples == 0)
	{
		std::cout << "- Not enough images to train (at least 2 are needed).\n";
		return stats;
	}

	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
	std::ostringstream summary;
	summary << std::fixed;

	if (bp_flag)
	{
		summary << "Epoch: " << epochs_trained + 1 << " - Perdida: " << std::setprecision(4) << stats.loss
			<< " - Accuracy: " << std::setprecision(2) << stats.accuracy * 100.0 << "%"
			<< " - LR: " << std::setprecision(6) << current_learning_rate()
			<< " - " << std::setprecision(1) << seconds << "s\n\n";

		++epochs_trained;
	}
	else
	{
		summary << "- Accuracy: " << std::setprecision(2) << stats.accuracy * 100.0 << "% +/- " << stats.accuracy_error() * 100.0 << "% - Perdida: " << std::setprecision(4) << stats.loss
			<< " (" << stats.samples << " images) -\n";
	}

	std::cout << summary.str();

	return stats;
}

EpochStats neural_network::run_pass(const ImageSet& data, bool training, bool prob_flag, bool report)
{
	EpochStats stats;

	if (layers.empty() || data.empty()) return stats;

	const size_t total = data.size();
	const int batch = training ? batch_size : std::max(batch_size, evaluation_batch);

	std::vector<size_t> order(total);
	std::iota(order.begin(), order.end(), size_t{ 0 });

	uint64_t epoch_seed = 0;

	if (training)
	{
		shuffle_indices(order, rng);

		const uint64_t high = rng();
		const uint64_t low = rng();
		epoch_seed = (high << 32) | low;

		if (augment.enabled && augmenter_.config() != augment)
		{
			const uint32_t seed = static_cast<uint32_t>(rng());
			augmenter_ = Augmenter(augment, seed);
		}
	}

	const bool augmenting = training && augment.enabled;

	std::vector<float> input(static_cast<size_t>(batch) * input_size);
	std::vector<uint8_t> labels(batch);
	std::vector<int> predictions(batch);
	std::vector<float> probs(static_cast<size_t>(batch) * num_classes);
	std::vector<float> gradient(training ? static_cast<size_t>(batch) * num_classes : 0);

	const float inv_std = 1.f / input_std;
	double loss_sum = 0.0;
	double window_loss = 0.0;
	size_t window_samples = 0;
	size_t correct = 0;
	size_t processed = 0;
	size_t last_report = 0;

	for (size_t start = 0; start < total; start += batch)
	{
		const int current = static_cast<int>(std::min<size_t>(static_cast<size_t>(batch), total - start));

		if (training && (current < 2 || (current < batch / 2 && processed > 0))) break;

		for (int s = 0; s < current; ++s)
		{
			const size_t pixels_count = data[order[start + s]].first.size();

			if (pixels_count != static_cast<size_t>(input_size))
			{
				std::cout << "- Error: an image has " << pixels_count << " pixels, expected " << input_size << ".\n";
				stats.diverged = true;
				return stats;
			}
		}

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(augmenting && current >= 16)
#endif
		for (int s = 0; s < current; ++s)
		{
			const Sample& sample = data[order[start + s]];
			const float* pixels = sample.first.data();
			float warped[input_size];

			if (augmenting)
			{
				augmenter_.apply(pixels, warped, mix64(epoch_seed + start + static_cast<size_t>(s)));
				pixels = warped;
			}

			float* row = input.data() + static_cast<size_t>(s) * input_size;

			for (int k = 0; k < input_size; ++k) row[k] = (pixels[k] - input_mean) * inv_std;

			labels[s] = sample.second;
		}

		const std::vector<float>& logits = forward(input.data(), current, training);
		const double batch_loss = softmax_cross_entropy(logits.data(), current, num_classes, labels.data(),
			probs.data(), training ? gradient.data() : nullptr, predictions.data(), training ? label_smoothing : 0.f);

		if (!std::isfinite(batch_loss))
		{
			std::cout << "- Error: the loss is not finite (sample " << processed << "), stopping.\n";
			stats.diverged = true;
			break;
		}

		loss_sum += batch_loss;
		window_loss += batch_loss;
		window_samples += current;

		for (int s = 0; s < current; ++s)
		{
			if (predictions[s] == labels[s]) ++correct;

			++stats.confusion[labels[s]][predictions[s]];

			if (prob_flag) std::cout << format_probabilities(probs.data() + static_cast<size_t>(s) * num_classes, predictions[s], labels[s]);
		}

		processed += current;

		if (training)
		{
			epoch_progress_ = static_cast<float>(processed) / static_cast<float>(total);

			backward(gradient.data());
			apply_gradients();

			if (report && processed - last_report >= report_interval && processed < total)
			{
				std::cout << "  [" << processed << "/" << total << "] Perdida: " << std::fixed << std::setprecision(4)
					<< window_loss / static_cast<double>(window_samples) << std::defaultfloat << '\n';

				last_report = processed;
				window_loss = 0.0;
				window_samples = 0;
			}
		}
	}

	if (training) epoch_progress_ = 0.f;

	stats.samples = processed;

	if (processed > 0)
	{
		stats.loss = loss_sum / static_cast<double>(processed);
		stats.accuracy = static_cast<double>(correct) / static_cast<double>(processed);
	}

	return stats;
}

bool neural_network::serialize(const char* file_name) const
{
	if (layers.empty())
	{
		std::cout << "- Error: there is no model to save.\n";
		return false;
	}

	for (const Layer& layer : layers)
	{
		if (!layer.is_finite())
		{
			std::cout << "- Error: the model contains non-finite values, it was not saved.\n";
			return false;
		}
	}

	json data;
	data["version"] = 2;
	data["input_mean"] = compact(input_mean);
	data["input_std"] = compact(input_std);
	data["epochs_trained"] = epochs_trained;
	data["holdout_every"] = holdout_every;
	data["layers"] = json::array();

	for (const Layer& layer : layers)
	{
		json node;
		node["in"] = layer.in_features;
		node["out"] = layer.out_features;
		node["relu"] = layer.ReLU;
		node["batchnorm"] = layer.batchnorm;
		node["dropout"] = compact(layer.dropout);
		node["weights"] = to_json(layer.weights.w);
		node["bias"] = to_json(layer.bias.w);

		if (layer.batchnorm)
		{
			node["gamma"] = to_json(layer.gamma.w);
			node["beta"] = to_json(layer.beta.w);
			node["running_mean"] = to_json(layer.running_mean);
			node["running_var"] = to_json(layer.running_var);
		}

		data["layers"].push_back(std::move(node));
	}

	const std::string temp_name = std::string(file_name) + ".tmp";

	{
		std::ofstream save_model(temp_name, std::ios::binary | std::ios::trunc);

		if (!save_model)
		{
			std::cout << "- Error: could not write " << temp_name << ".\n";
			return false;
		}

		save_model << data.dump();
		save_model.flush();

		if (!save_model)
		{
			std::cout << "- Error: could not write " << temp_name << ".\n";
			save_model.close();
			std::error_code ignored;
			std::filesystem::remove(temp_name, ignored);
			return false;
		}
	}

	std::error_code error;
	std::filesystem::rename(temp_name, file_name, error);

	if (error)
	{
		std::cout << "- Error: could not replace " << file_name << ": " << error.message() << ".\n";
		std::filesystem::remove(temp_name, error);
		return false;
	}

	return true;
}

bool neural_network::deserialize(const char* file_name, bool create_if_missing)
{
	layers.clear();
	step_count = 0;

	std::error_code error;

	if (!std::filesystem::exists(file_name, error) || std::filesystem::file_size(file_name, error) == 0)
	{
		if (!create_if_missing)
		{
			std::cout << "- Error: there is no trained model in " << file_name << ", train one first.\n";
			return false;
		}

		input_mean = 0.1307f;
		input_std = 0.3081f;
		epochs_trained = 0;
		holdout_every = 0;
		build_default_layers();

		return true;
	}

	try
	{
		std::ifstream ifstream(file_name);

		if (!ifstream) throw std::runtime_error("the file cannot be opened");

		json model;
		ifstream >> model;
		ifstream.close();

		const bool legacy = !model.contains("version");
		const int version = legacy ? 1 : model.at("version").get<int>();

		if (version > 2) throw std::runtime_error("unsupported model version " + std::to_string(version));

		const json& nodes = model.at("layers");

		if (!nodes.is_array() || nodes.empty()) throw std::runtime_error("the model has no layers");

		std::vector<Layer> loaded;
		loaded.reserve(nodes.size());

		for (size_t i = 0; i < nodes.size(); ++i)
		{
			const json& node = nodes[i];

			try
			{
				if (legacy)
				{
					const json& neurons = node.at("neurons");

					if (neurons.empty()) throw std::runtime_error("the layer has no neurons");

					const int out = static_cast<int>(neurons.size());
					const int in = static_cast<int>(neurons[0].at("weights").size());

					Layer layer(in, out, i + 1 < nodes.size(), false, 0.f);

					for (int j = 0; j < out; ++j)
					{
						const std::vector<float> row = read_floats(neurons[j], "weights", static_cast<size_t>(in));

						std::copy(row.begin(), row.end(), layer.weights.w.begin() + static_cast<size_t>(j) * in);
						layer.bias.w[j] = neurons[j].at("bias").get<float>();
					}

					loaded.push_back(std::move(layer));
				}
				else
				{
					const int in = node.at("in").get<int>();
					const int out = node.at("out").get<int>();

					Layer layer(in, out, node.at("relu").get<bool>(), node.at("batchnorm").get<bool>(), node.at("dropout").get<float>());

					layer.weights.w = read_floats(node, "weights", static_cast<size_t>(in) * out);
					layer.bias.w = read_floats(node, "bias", static_cast<size_t>(out));

					if (layer.batchnorm)
					{
						layer.gamma.w = read_floats(node, "gamma", static_cast<size_t>(out));
						layer.beta.w = read_floats(node, "beta", static_cast<size_t>(out));
						layer.running_mean = read_floats(node, "running_mean", static_cast<size_t>(out));
						layer.running_var = read_floats(node, "running_var", static_cast<size_t>(out));
					}

					loaded.push_back(std::move(layer));
				}
			}
			catch (const std::exception& e)
			{
				throw std::runtime_error("layer " + std::to_string(i) + ": " + e.what());
			}
		}

		if (loaded.front().in_features != input_size) throw std::runtime_error("the first layer expects " + std::to_string(loaded.front().in_features) + " inputs, not " + std::to_string(input_size));
		if (loaded.back().out_features != num_classes) throw std::runtime_error("the last layer has " + std::to_string(loaded.back().out_features) + " outputs, not " + std::to_string(num_classes));

		for (size_t i = 0; i < loaded.size(); ++i)
		{
			if (i > 0 && loaded[i].in_features != loaded[i - 1].out_features) throw std::runtime_error("layer " + std::to_string(i) + " does not match the size of the previous layer");
			if (!loaded[i].is_finite()) throw std::runtime_error("layer " + std::to_string(i) + " has non-finite values");
		}

		input_mean = legacy ? 0.f : model.value("input_mean", 0.f);
		input_std = legacy ? 1.f : model.value("input_std", 1.f);
		epochs_trained = legacy ? 0 : model.value("epochs_trained", 0);
		holdout_every = legacy ? 0 : model.value("holdout_every", size_t{ 0 });

		if (!(input_std > 0.f)) throw std::runtime_error("input_std must be positive");

		layers = std::move(loaded);
	}
	catch (const std::exception& e)
	{
		layers.clear();
		std::cout << "- Error: could not load " << file_name << ": " << e.what() << "\n";
		return false;
	}

	return true;
}

bool neural_network::read_images(const std::string& type)
{
	const std::string image_path = "./dataset/images/" + type + "-images.idx3-ubyte";
	const std::string label_path = "./dataset/labels/" + type + "-labels.idx1-ubyte";

	std::ifstream file(image_path, std::ios::binary);
	std::ifstream file2(label_path, std::ios::binary);

	if (!file || !file2)
	{
		std::cout << "- Error: could not open " << (!file ? image_path : label_path) << "\n";
		return false;
	}

	const uint32_t magic = read_uint32(file);
	const uint32_t num_images = read_uint32(file);
	const uint32_t rows = read_uint32(file);
	const uint32_t cols = read_uint32(file);

	const uint32_t magic_l = read_uint32(file2);
	const uint32_t num_labels = read_uint32(file2);

	if (!file || !file2 || magic != 0x00000803 || magic_l != 0x00000801 || num_images != num_labels || num_images > max_images ||
		static_cast<uint64_t>(rows) * cols != static_cast<uint64_t>(input_size))
	{
		std::cout << "- Error: the dataset files for '" << type << "' are not valid MNIST IDX files (28x28 images).\n";
		return false;
	}

	std::vector<uint8_t> raw_pixels(static_cast<size_t>(num_images) * input_size);
	std::vector<uint8_t> raw_labels(num_images);

	file.read(reinterpret_cast<char*>(raw_pixels.data()), static_cast<std::streamsize>(raw_pixels.size()));
	file2.read(reinterpret_cast<char*>(raw_labels.data()), static_cast<std::streamsize>(raw_labels.size()));

	if (!file || !file2)
	{
		std::cout << "- Error: the dataset files for '" << type << "' are truncated.\n";
		return false;
	}

	for (uint8_t label : raw_labels)
	{
		if (label >= num_classes)
		{
			std::cout << "- Error: the dataset files for '" << type << "' contain an invalid label.\n";
			return false;
		}
	}

	images.reserve(images.size() + num_images);

	for (size_t i = 0; i < num_images; ++i)
	{
		const uint8_t* source = raw_pixels.data() + i * input_size;

		std::vector<float> pixels(input_size);

		for (int j = 0; j < input_size; ++j)
		{
			pixels[j] = source[j] / 255.f;
		}

		images.push_back({ std::move(pixels), raw_labels[i] });
	}

	return true;
}
