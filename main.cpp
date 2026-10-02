/*

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <array>
#include <random>
#include <nlohmann/json.hpp>

class Neuron
{
public:
	std::vector<float> weights;
	std::vector<float> inputs;
	float bias;
	float learning_rate;
	float output_pre_ReLU;

	Neuron() : weights({}), bias(0.f), learning_rate(0.01f), inputs({}), output_pre_ReLU(0.f) {}

	void init(size_t inputs_size, std::normal_distribution<float>& dist)
	{
		static std::mt19937 gen(std::random_device{}());

		for (int i = 0; i < inputs_size; ++i)
		{
			weights.push_back(dist(gen));
		}
	}

	void update_parameters(float logit_gradient)
	{
		for (int i = 0; i < weights.size(); ++i)
		{
			float weight_gradient = logit_gradient * inputs[i];
			weights[i] -= weight_gradient * learning_rate;
		}

		bias -= logit_gradient * learning_rate;
	}

	float get_output(std::vector<float>& inputs, std::normal_distribution<float>& dist, bool ReLU = true)
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
		if(ReLU) output = output >= 0 ? output : 0; // ReLU
		
		return output;
	}
};

uint32_t read_uint32(std::ifstream& file)
{
	uint8_t bytes[4];
	file.read(reinterpret_cast<char*>(bytes), 4);

	return (bytes[0] << 24) |
		(bytes[1] << 16) |
		(bytes[2] << 8) |
		bytes[3];
}

void read_images(const std::string& type, std::vector<std::pair<std::vector<float>, uint8_t>>& images)
{
	std::ifstream file("./dataset/images/"+type+"-images.idx3-ubyte", std::ios::binary);
	std::ifstream file2("./dataset/labels/"+type+"-labels.idx1-ubyte", std::ios::binary);

	uint32_t magic = read_uint32(file);
	uint32_t numImages = read_uint32(file);
	uint32_t cols = read_uint32(file);
	uint32_t rows = read_uint32(file); // para avanzar el cursor

	uint32_t magic_l = read_uint32(file2);
	uint32_t numLabels = read_uint32(file2); // para avanzar el cursor

	int image_size = cols * rows;
	images.reserve(numImages);

	for (size_t i = 0; i < numImages; ++i)
	{
		std::vector<uint8_t> raw_pixels(image_size);
		uint8_t label = 0;

		file.read(reinterpret_cast<char*>(raw_pixels.data()), image_size);
		file2.read(reinterpret_cast<char*>(&label), 1);

		std::vector<float> pixels;
		pixels.reserve(raw_pixels.size());

		for (int j = 0; j < raw_pixels.size(); ++j)
		{
			pixels.push_back(raw_pixels[j] / 255.f); // normalizing
		}

		images.push_back({ std::move(pixels), label });
	}
}

class Layer
{
public:
	std::vector<Neuron> neurons;
	bool ReLU;
	std::vector<float> gradients;
	std::vector<float> outputs;

	Layer(size_t layer_size, std::vector<Neuron>* neurons = nullptr, bool ReLU = true) : ReLU(ReLU)
	{
		if (neurons == nullptr) this->neurons.resize(layer_size);
		else this->neurons = std::move(*neurons);
	}

	void gradient(std::vector<float>& next_gradients, std::vector<Neuron>& next_neurons)
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

	void last_layer_gradient(std::vector<float>& probs, int label)
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

	void start(std::vector<float> inputs)
	{
		float stddev = std::sqrt(2.0f / inputs.size());
		std::normal_distribution<float> dist(0.0f, stddev);

		outputs.clear();
		for (size_t i = 0; i < neurons.size(); ++i)
		{
			outputs.push_back(neurons[i].get_output(inputs, dist, ReLU));
		}
	}
};

std::pair<float, float> get_biggest(std::vector<float>& nums)
{
	float biggest = std::numeric_limits<float>::lowest();
	int index = 0;

	for (int i = 0; i < nums.size(); ++i)
	{
		if (nums[i] > biggest)
		{
			biggest = nums[i];
			index = i;
		}
	}
	return {biggest, index};
}

std::vector<float> apply_stable_softmax(std::vector<float>& logits)
{
	float accum = 0;
	float biggest = get_biggest(logits).first;
	std::vector<float> probs = {};

	for (float& logit : logits) accum += std::exp(logit - biggest);

	for (float& logit : logits) probs.push_back(std::exp(logit - biggest)/accum);

	return probs;
}

void serialize(const char* file_name, std::vector<Layer>& layers)
{
	// serializar los pesos
	nlohmann::json data;

	for (int i = 0; i < layers.size(); ++i)
	{
		for (int j = 0; j < layers[i].neurons.size(); ++j)
		{
			data["layers"][i]["neurons"][j]["weights"] = layers[i].neurons[j].weights;
			data["layers"][i]["neurons"][j]["bias"] = layers[i].neurons[j].bias;
		}
	}

	std::ofstream save_model(file_name);

	save_model << data.dump(4);
}

void deserialize(const char* file_name, std::vector<Layer>& layers)
{
	std::ifstream ifstream(file_name);

	nlohmann::json model;
	ifstream >> model;

	if (model.empty()) layers = { Layer(128), Layer(64), Layer(32), Layer(10, nullptr, false) };
	else
	{
		for (int i = 0; i < model["layers"].size(); ++i)
		{
			size_t neurons_size = model["layers"][i]["neurons"].size();
			std::vector<Neuron> neurons;

			for (int j = 0; j < neurons_size; ++j)
			{
				Neuron n;
				n.weights = model["layers"][i]["neurons"][j]["weights"].get<std::vector<float>>();
				n.bias = model["layers"][i]["neurons"][j]["bias"].get<float>();

				neurons.push_back(std::move(n));
			}

			layers.push_back({ neurons_size, &neurons, i != model["layers"].size() - 1});
		}
	}

	ifstream.close();
}


int main()
{
	std::vector<Layer> layers;

	deserialize("model.json", layers);

	std::vector<std::pair<std::vector<float>, uint8_t>> images;

	read_images("train", images);

	int iter = 0;
	float average_loss = 0.f;
	int corrects = 0;

	while (iter < images.size())
	{
		layers[0].start(images[iter].first);

		for (size_t i = 1; i < layers.size(); ++i)
		{
			layers[i].start(layers[i - 1].outputs);
		}

		std::vector<float> probs = apply_stable_softmax(layers.back().outputs);

		
		std::cout << "[ ";
		for (float& prob : probs)
		{
			std::cout << prob*100 << "%" << ", ";
		}
		std::cout << " ] - Real Number: " << static_cast<int>(images[iter].second) << '\n';

		float loss = -std::log(probs[images[iter].second]);
		average_loss += loss;

		if (iter != 0 && iter % 1000 == 0)
		{
			std::cout << "Epoch: " << iter / 1000 << " - Perdida: " << average_loss / 1000.f << "\n\n";
			average_loss = 0;
		}

		if (get_biggest(probs).second == images[iter].second) ++corrects;

		layers.back().last_layer_gradient(probs, images[iter].second);

		for (int i = layers.size() - 2; i >= 0; --i)
		{
			layers[i].gradient(layers[i + 1].gradients, layers[i + 1].neurons);
		}

		for (int i = layers.size() - 1; i >= 0; --i)
		{
			for (int j = 0; j < layers[i].neurons.size(); ++j)
			{
				layers[i].neurons[j].update_parameters(layers[i].gradients[j]);
			}
		}
		
		++iter;
	}

	std::cout << "- Accuracy: " << corrects / images.size() * 100.f << "% -" << '\n';

	serialize("model.json", layers);

	return 0;
}

*/