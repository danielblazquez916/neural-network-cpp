#include "state.hpp"
#include "renderer.hpp"

State::State(int batch_size, float learning_rate) : nn(batch_size, learning_rate), loaded_dataset(Dataset::None)
{
	// 3 states
	states[state_type::Training] = [this]() { training(); };

	states[state_type::Testing] = [this]() { testing(); };

	states[state_type::Drawing] = [this]() { drawing(); };
}

void State::swap(state_type new_state)
{
	std::cout << "-------------------------------\n\n";

	states[new_state]();
}

void State::drawing()
{
	std::cout << "- Drawing Mode -\n\n" << std::flush;

	nn.layers.clear();
	nn.deserialize("model.json");

	nn.images.clear();

	loaded_dataset = Dataset::None;

	render_loop(nn);

	nn.process(1, 0); // 1 = show probabilities | 0 = backpropagation //
}

void State::training()
{
	std::cout << "- Training Mode -\n\n";

	nn.layers.clear();
	nn.deserialize("model.json");

	if (loaded_dataset != Dataset::Train)
	{
		nn.images.clear();
		nn.read_images("train");

		loaded_dataset = Dataset::Train;
	}

	nn.process(0, 1);

	nn.serialize("model.json");
}

void State::testing()
{
	std::cout << "- Testing Mode -\n\n";

	nn.layers.clear();
	nn.deserialize("model.json");

	if (loaded_dataset != Dataset::Test)
	{
		nn.images.clear();
		nn.read_images("t10k");
		loaded_dataset = Dataset::Test;
	}

	nn.process(0, 0);
}
