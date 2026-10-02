#include <iostream>
#include "neural_network.hpp"
#include <unordered_map>
#include <functional>
#include <optional>

enum class state_type
{
	Training,
	Testing,
	Drawing
};

enum class Dataset
{
	None,
	Train,
	Test
};

class State
{
public:
	neural_network nn;
	std::unordered_map<state_type, std::function<void()>> states;
	Dataset loaded_dataset;

	State(int batch_size, float learning_rate);
	~State() = default;

	void swap(state_type new_state);

	// all states
	void drawing();

	void training();

	void testing();
};