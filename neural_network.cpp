#include "neural_network.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>

neural_network::neural_network(int batch_size, float learning_rate) : iter(0), average_loss(0.f), corrects(0), batch_size(batch_size), learning_rate(learning_rate) {};

std::vector<float> neural_network::apply_stable_softmax(std::vector<float>& logits)
{
	float accum = 0;
	float biggest = get_biggest(logits).first;
	std::vector<float> probs = {};

	for (float& logit : logits) accum += std::exp(logit - biggest);

	for (float& logit : logits) probs.push_back(std::exp(logit - biggest) / accum);

	return probs;
}

void neural_network::backprop(bool flag, std::vector<float>& probs)
{
	if (!flag) return;

	layers.back().last_layer_gradient(probs, images[iter].second);

	for (int i = layers.size() - 2; i >= 0; --i)
	{
		layers[i].gradient(layers[i + 1].gradients, layers[i + 1].neurons);
	}

	for (int i = layers.size() - 1; i >= 0; --i)
	{
		for (int j = 0; j < layers[i].neurons.size(); ++j)
		{
			// layers[i].neurons[j].update_parameters(layers[i].gradients[j]);
			layers[i].neurons[j].batching(layers[i].gradients[j], iter, batch_size, images.size(), learning_rate);
		}
	}
}

void neural_network::serialize(const char* file_name)
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

void neural_network::deserialize(const char* file_name)
{
	std::ifstream ifstream(file_name);

	if (ifstream.peek() == std::ifstream::traits_type::eof())
	{
		if (!layers.empty()) return;

		layers = { Layer(256), Layer(128), Layer(64), Layer(10, nullptr, false) };
		return;
	}

	nlohmann::json model;
	ifstream >> model;

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

		layers.push_back({ neurons_size, &neurons, i != model["layers"].size() - 1 });
	}

	ifstream.close();
}

void neural_network::probability(bool flag, std::vector<float>& probs)
{
	if (!flag) return;

	std::cout << "[ ";
	for (float& prob : probs)
	{
		std::cout << prob * 100 << "%" << ", ";
	}
	std::cout << " ] - Real Number: " << static_cast<int>(images[iter].second) << std::endl;
}

void neural_network::read_images(const std::string& type)
{
	std::ifstream file("./dataset/images/" + type + "-images.idx3-ubyte", std::ios::binary);
	std::ifstream file2("./dataset/labels/" + type + "-labels.idx1-ubyte", std::ios::binary);

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

void neural_network::shuffle()
{
	static std::random_device rd;
	static std::mt19937 gen(rd());

	std::shuffle(images.begin(), images.end(), gen);
}

void neural_network::restart_values()
{
	iter = 0; 
	average_loss = 0.f; 
	corrects = 0;
}

void neural_network::process(bool prob_flag, bool bp_flag)
{
	while (iter < images.size())
	{
		layers[0].start(images[iter].first);

		for (size_t i = 1; i < layers.size(); ++i)
		{
			layers[i].start(layers[i - 1].outputs);
		}

		std::vector<float> probs = apply_stable_softmax(layers.back().outputs);

		probability(prob_flag, probs);

		float loss = -std::log(probs[images[iter].second]);
		average_loss += loss;

		if (iter != 0 && iter % 1000 == 0)
		{
			shuffle();

			std::cout << "Epoch: " << iter / 1000 << " - Perdida: " << average_loss / 1000.f << "\n\n";
			average_loss = 0;
		}

		if (get_biggest(probs).second == images[iter].second) ++corrects;

		backprop(bp_flag, probs);

		++iter;
	}

	std::cout << "- Accuracy: " << static_cast<float>(corrects) / images.size() * 100.f << "% -" << '\n';
	restart_values();
}

