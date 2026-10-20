#include "digit_preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace
{
	constexpr int canvas_size = 28;
	constexpr int digit_size = 20;
	constexpr double canvas_center = (canvas_size - 1) / 2.0;

	using Weights = std::vector<std::pair<int, double>>;

	Weights resampling_weights(int destination, int source_size, int target_size)
	{
		Weights weights;
		const double ratio = static_cast<double>(source_size) / target_size;
		if (ratio > 1.0)
		{
			const double begin = destination * ratio;
			const double end = (destination + 1) * ratio;
			for (int i = static_cast<int>(std::floor(begin));
				i < std::min(source_size, static_cast<int>(std::ceil(end))); ++i)
			{
				const double overlap = std::min(end, i + 1.0) - std::max(begin, static_cast<double>(i));
				weights.emplace_back(i, overlap / ratio);
			}
		}
		else
		{
			const double position = std::clamp((destination + 0.5) * ratio - 0.5, 0.0, source_size - 1.0);
			const int left = static_cast<int>(std::floor(position));
			const double fraction = position - left;
			weights.emplace_back(left, 1.0 - fraction);
			weights.emplace_back(std::min(left + 1, source_size - 1), fraction);
		}
		return weights;
	}

	double sample(const std::vector<double>& pixels, double x, double y)
	{
		const int left = static_cast<int>(std::floor(x));
		const int top = static_cast<int>(std::floor(y));
		const double fraction_x = x - left;
		const double fraction_y = y - top;
		const auto at = [&](int col, int row)
			{
				return col >= 0 && row >= 0 && col < canvas_size && row < canvas_size
					? pixels[row * canvas_size + col] : 0.0;
			};
		return (1.0 - fraction_y) * ((1.0 - fraction_x) * at(left, top) + fraction_x * at(left + 1, top))
			+ fraction_y * ((1.0 - fraction_x) * at(left, top + 1) + fraction_x * at(left + 1, top + 1));
	}

}

sf::Image preprocess_digit(const sf::Image& image)
{
	sf::Image result({ canvas_size, canvas_size }, sf::Color::Black);
	const auto size = image.getSize();
	unsigned int min_x = size.x, min_y = size.y, max_x = 0, max_y = 0;

	for (unsigned int y = 0; y < size.y; ++y)
	{
		for (unsigned int x = 0; x < size.x; ++x)
		{
			if (image.getPixel({ x, y }).r == 0) continue;
			min_x = std::min(min_x, x);
			min_y = std::min(min_y, y);
			max_x = std::max(max_x, x);
			max_y = std::max(max_y, y);
		}
	}
	if (min_x == size.x || min_y == size.y) return result;

	const int width = static_cast<int>(max_x - min_x + 1);
	const int height = static_cast<int>(max_y - min_y + 1);
	const double scale = static_cast<double>(digit_size) / std::max(width, height);
	const int target_width = std::max(1, static_cast<int>(std::lround(width * scale)));
	const int target_height = std::max(1, static_cast<int>(std::lround(height * scale)));
	const int offset_x = (canvas_size - target_width) / 2;
	const int offset_y = (canvas_size - target_height) / 2;

	std::vector<Weights> horizontal_weights;
	for (int x = 0; x < target_width; ++x)
		horizontal_weights.push_back(resampling_weights(x, width, target_width));

	std::vector<double> canvas(canvas_size * canvas_size, 0.0);
	double mass = 0.0, moment_x = 0.0, moment_y = 0.0;
	int ink_left = canvas_size, ink_top = canvas_size, ink_right = 0, ink_bottom = 0;
	for (int y = 0; y < target_height; ++y)
	{
		const auto vertical_weights = resampling_weights(y, height, target_height);
		for (int x = 0; x < target_width; ++x)
		{
			double value = 0.0;
			for (const auto& [source_y, weight_y] : vertical_weights)
				for (const auto& [source_x, weight_x] : horizontal_weights[x])
					value += image.getPixel({ min_x + source_x, min_y + source_y }).r / 255.0 * weight_x * weight_y;

			const int col = x + offset_x, row = y + offset_y;
			canvas[row * canvas_size + col] = value;
			mass += value;
			moment_x += col * value;
			moment_y += row * value;
			if (value > 0.0)
			{
				ink_left = std::min(ink_left, col);
				ink_top = std::min(ink_top, row);
				ink_right = std::max(ink_right, col);
				ink_bottom = std::max(ink_bottom, row);
			}
		}
	}
	if (mass == 0.0) return result;

	const double shift_x = std::clamp(canvas_center - moment_x / mass, -static_cast<double>(ink_left), canvas_size - 1.0 - ink_right);
	const double shift_y = std::clamp(canvas_center - moment_y / mass, -static_cast<double>(ink_top), canvas_size - 1.0 - ink_bottom);

	for (unsigned int y = 0; y < canvas_size; ++y)
	{
		for (unsigned int x = 0; x < canvas_size; ++x)
		{
			const auto value = static_cast<std::uint8_t>(std::lround(
				std::clamp(sample(canvas, x - shift_x, y - shift_y), 0.0, 1.0) * 255.0 + 1e-9));
			result.setPixel({ x, y }, sf::Color(value, value, value));
		}
	}
	return result;
}

std::vector<float> image_to_vector(const sf::Image& image)
{
	if (image.getSize() != sf::Vector2u{ canvas_size, canvas_size })
		throw std::invalid_argument("image_to_vector requires a preprocessed 28x28 image");

	std::vector<float> input;
	input.reserve(canvas_size * canvas_size);
	for (unsigned int y = 0; y < canvas_size; ++y)
		for (unsigned int x = 0; x < canvas_size; ++x)
			input.push_back(image.getPixel({ x, y }).r / 255.0f);
	return input;
}
