#include "drawing.hpp"
#include <algorithm>
#include <cmath>

namespace
{
	constexpr int image_size = 28;
	constexpr int box_size = 19;
	constexpr float blur_sigma = 0.8f;
	constexpr float ink_gain = 1.3f;
	constexpr float center_of_mass = 14.f;
	constexpr int minimum_side = 4;

	std::vector<float> area_weights(int n_in, int n_out)
	{
		std::vector<float> weights(static_cast<size_t>(n_out) * n_in, 0.f);
		const float scale = static_cast<float>(n_in) / static_cast<float>(n_out);

		for (int o = 0; o < n_out; ++o)
		{
			const float begin = static_cast<float>(o) * scale;
			const float end = static_cast<float>(o + 1) * scale;
			const int first = static_cast<int>(std::floor(begin + 1e-6f));
			const int last = std::min(n_in, static_cast<int>(std::ceil(end - 1e-6f)));

			for (int i = first; i < last; ++i)
			{
				const float overlap = std::min(end, static_cast<float>(i + 1)) - std::max(begin, static_cast<float>(i));

				weights[static_cast<size_t>(o) * n_in + i] = std::max(0.f, overlap) / scale;
			}
		}

		return weights;
	}

	void blur(std::vector<float>& image, float sigma)
	{
		const int radius = static_cast<int>(4.f * sigma + 0.5f);
		std::vector<float> kernel(2 * radius + 1);
		float sum = 0.f;

		for (int i = -radius; i <= radius; ++i)
		{
			kernel[i + radius] = std::exp(-0.5f * static_cast<float>(i * i) / (sigma * sigma));
			sum += kernel[i + radius];
		}

		for (float& k : kernel) k /= sum;

		std::vector<float> temp(image.size(), 0.f);

		for (int y = 0; y < image_size; ++y)
		{
			for (int x = 0; x < image_size; ++x)
			{
				float acc = 0.f;

				for (int i = -radius; i <= radius; ++i)
				{
					const int xx = x + i;

					if (xx >= 0 && xx < image_size) acc += kernel[i + radius] * image[y * image_size + xx];
				}

				temp[y * image_size + x] = acc;
			}
		}

		for (int y = 0; y < image_size; ++y)
		{
			for (int x = 0; x < image_size; ++x)
			{
				float acc = 0.f;

				for (int i = -radius; i <= radius; ++i)
				{
					const int yy = y + i;

					if (yy >= 0 && yy < image_size) acc += kernel[i + radius] * temp[yy * image_size + x];
				}

				image[y * image_size + x] = acc;
			}
		}
	}
}

bool preprocess_drawing(const std::vector<float>& grid, int rows, int cols, std::vector<float>& out)
{
	if (rows <= 0 || cols <= 0 || grid.size() != static_cast<size_t>(rows) * cols) return false;

	int top = rows, bottom = -1, left = cols, right = -1;

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			if (grid[static_cast<size_t>(r) * cols + c] > 0.f)
			{
				top = std::min(top, r);
				bottom = std::max(bottom, r);
				left = std::min(left, c);
				right = std::max(right, c);
			}
		}
	}

	if (bottom < 0) return false;

	const int height = bottom - top + 1;
	const int width = right - left + 1;

	if (std::max(height, width) < minimum_side) return false;

	const double scale = static_cast<double>(box_size) / static_cast<double>(std::max(height, width));
	const int new_height = std::max(1, static_cast<int>(std::nearbyint(static_cast<double>(height) * scale)));
	const int new_width = std::max(1, static_cast<int>(std::nearbyint(static_cast<double>(width) * scale)));

	const std::vector<float> row_weights = area_weights(height, new_height);
	const std::vector<float> col_weights = area_weights(width, new_width);

	std::vector<float> image(static_cast<size_t>(image_size) * image_size, 0.f);
	const int offset_y = (image_size - new_height) / 2;
	const int offset_x = (image_size - new_width) / 2;

	for (int y = 0; y < new_height; ++y)
	{
		for (int x = 0; x < new_width; ++x)
		{
			float value = 0.f;

			for (int i = 0; i < height; ++i)
			{
				const float wy = row_weights[static_cast<size_t>(y) * height + i];

				if (wy == 0.f) continue;

				for (int j = 0; j < width; ++j)
				{
					const float cell = grid[static_cast<size_t>(top + i) * cols + left + j] > 0.f ? 1.f : 0.f;

					value += wy * col_weights[static_cast<size_t>(x) * width + j] * cell;
				}
			}

			image[static_cast<size_t>(offset_y + y) * image_size + offset_x + x] = value;
		}
	}

	blur(image, blur_sigma);

	const float peak = std::max(*std::max_element(image.begin(), image.end()), 1e-6f);

	for (float& value : image) value = std::min(1.f, ink_gain * value / peak);

	double mass = 0.0, mass_y = 0.0, mass_x = 0.0;

	for (int y = 0; y < image_size; ++y)
	{
		for (int x = 0; x < image_size; ++x)
		{
			const double value = image[static_cast<size_t>(y) * image_size + x];

			mass += value;
			mass_y += value * y;
			mass_x += value * x;
		}
	}

	if (!(mass > 0.0)) return false;

	const int shift_y = static_cast<int>(std::nearbyint(center_of_mass - mass_y / mass));
	const int shift_x = static_cast<int>(std::nearbyint(center_of_mass - mass_x / mass));

	out.assign(image.size(), 0.f);

	for (int y = 0; y < image_size; ++y)
	{
		for (int x = 0; x < image_size; ++x)
		{
			const int source_y = y - shift_y;
			const int source_x = x - shift_x;

			if (source_y >= 0 && source_y < image_size && source_x >= 0 && source_x < image_size)
			{
				out[static_cast<size_t>(y) * image_size + x] = image[static_cast<size_t>(source_y) * image_size + source_x];
			}
		}
	}

	return true;
}

std::string ascii_digit(const std::vector<float>& pixels)
{
	static const char shades[] = " .:-=+*#%@";
	std::string text;

	if (pixels.size() != static_cast<size_t>(image_size) * image_size) return text;

	for (int y = 0; y < image_size; ++y)
	{
		for (int x = 0; x < image_size; ++x)
		{
			const float value = std::min(1.f, std::max(0.f, pixels[static_cast<size_t>(y) * image_size + x]));

			text += shades[std::min(9, static_cast<int>(value * 9.99f))];
		}

		text += '\n';
	}

	return text;
}
