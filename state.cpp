#include "state.hpp"
#include "renderer.hpp"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace
{
	constexpr const char* model_file = "model.json";
	constexpr int max_epochs = 1000;
	constexpr int recommended_epochs = 60;
	constexpr size_t validation_every = 12;

	int ask_epochs()
	{
		int epochs = 0;

		while (true)
		{
			std::cout << "- How many epochs do you want to train? (1-" << max_epochs << ", " << recommended_epochs << " gives the best result)\n" << std::flush;

			if (std::cin >> epochs && epochs >= 1 && epochs <= max_epochs) return epochs;

			if (std::cin.eof()) return 0;

			std::cout << "- Try again, don't be stupid pls\n\n";

			std::cin.clear();
			std::cin.ignore(1000, '\n');
		}
	}

	std::string describe(const EpochStats& stats)
	{
		std::ostringstream text;
		text << std::fixed << "Perdida: " << std::setprecision(4) << stats.loss << " - Accuracy: " << std::setprecision(2) << stats.accuracy * 100.0 << "%";

		return text.str();
	}

	void print_details(const EpochStats& stats)
	{
		std::ostringstream text;
		text << std::fixed << std::setprecision(1) << "- Per digit:";

		for (int digit = 0; digit < EpochStats::classes; ++digit)
		{
			int total = 0;

			for (int c = 0; c < EpochStats::classes; ++c) total += stats.confusion[digit][c];

			if (total > 0) text << "  " << digit << ": " << 100.0 * stats.confusion[digit][digit] / total << "%";
		}

		text << "\n";

		std::vector<std::pair<int, std::pair<int, int>>> mistakes;

		for (int real = 0; real < EpochStats::classes; ++real)
		{
			for (int predicted = 0; predicted < EpochStats::classes; ++predicted)
			{
				if (real != predicted && stats.confusion[real][predicted] > 0) mistakes.push_back({ stats.confusion[real][predicted], { real, predicted } });
			}
		}

		std::sort(mistakes.begin(), mistakes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

		if (!mistakes.empty())
		{
			text << "- Most frequent mistakes (real -> predicted):";

			for (size_t i = 0; i < mistakes.size() && i < 5; ++i) text << "  " << mistakes[i].second.first << "->" << mistakes[i].second.second << " (" << mistakes[i].first << ")";

			text << "\n";
		}

		std::cout << text.str();
	}
}

State::State(int batch_size, float learning_rate) : nn(batch_size, learning_rate), loaded_dataset(Dataset::None)
{
	nn.augment.enabled = true;

	states[state_type::Training] = [this]() { training(); };

	states[state_type::Testing] = [this]() { testing(); };

	states[state_type::Drawing] = [this]() { drawing(); };
}

void State::swap(state_type new_state)
{
	std::cout << "-------------------------------\n\n";

	states[new_state]();
}

bool State::load_model(bool create_if_missing)
{
	nn.layers.clear();

	return nn.deserialize(model_file, create_if_missing);
}

bool State::load_dataset(Dataset wanted)
{
	if (loaded_dataset == wanted) return true;

	nn.images.clear();
	nn.validation.clear();
	loaded_dataset = Dataset::None;

	if (!nn.read_images(wanted == Dataset::Train ? "train" : "t10k")) return false;

	if (wanted == Dataset::Train) nn.hold_out_validation(validation_every);

	loaded_dataset = wanted;

	return true;
}

void State::drawing()
{
	std::cout << "- Drawing Mode -\n\n" << std::flush;

	if (!load_model(false)) return;

	nn.images.clear();
	nn.validation.clear();

	loaded_dataset = Dataset::None;

	render_loop(nn);

	if (nn.images.empty())
	{
		std::cout << "- Nothing was drawn.\n";
		return;
	}

	nn.process(1, 0);
}

void State::training()
{
	std::cout << "- Training Mode -\n\n";

	if (!load_model(true)) return;

	if (!load_dataset(Dataset::Train)) return;

	const int epochs = ask_epochs();

	if (epochs == 0) return;

	double best_loss = std::numeric_limits<double>::infinity();

	if (nn.holdout_every == validation_every)
	{
		best_loss = nn.evaluate(nn.validation).loss;

		std::cout << "- The current model scores " << std::fixed << std::setprecision(4) << best_loss << std::defaultfloat << " (validation loss).\n\n";
	}

	nn.holdout_every = validation_every;
	nn.begin_training_session(epochs);

	bool saved = false;

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		const EpochStats stats = nn.process(0, 1);

		if (stats.diverged)
		{
			std::cout << "- Training stopped. " << model_file << " was not modified by this epoch.\n";
			return;
		}

		if (stats.samples == 0) return;

		const EpochStats validation = nn.evaluate(nn.validation);

		std::cout << "  Validation - " << describe(validation);

		if (validation.loss < best_loss)
		{
			if (!nn.serialize(model_file))
			{
				std::cout << "\n- Training stopped. Could not save " << model_file << ".\n";
				return;
			}

			best_loss = validation.loss;
			saved = true;

			std::cout << "  (best so far, saved)";
		}

		std::cout << "\n\n";
	}

	if (saved) std::cout << "- Best model saved in " << model_file << "\n";
	else std::cout << "- No epoch improved the model, " << model_file << " was not modified.\n";
}

void State::testing()
{
	std::cout << "- Testing Mode -\n\n";

	if (!load_model(false)) return;

	if (!load_dataset(Dataset::Test)) return;

	const EpochStats stats = nn.process(0, 0);

	if (stats.samples > 0 && !stats.diverged) print_details(stats);
}
