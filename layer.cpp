#include "layer.hpp"
#include "kernels.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Layer::Layer() : in_features(0), out_features(0), ReLU(true), batchnorm(false), dropout(0.f), inputs_(nullptr), batch_(0), dropout_active_(false) {}

Layer::Layer(int in_features, int out_features, bool ReLU, bool batchnorm, float dropout)
	: in_features(in_features), out_features(out_features), ReLU(ReLU), batchnorm(batchnorm), dropout(dropout),
	  inputs_(nullptr), batch_(0), dropout_active_(false)
{
	if (in_features <= 0 || out_features <= 0) throw std::invalid_argument("Layer: sizes must be positive");
	if (dropout < 0.f || dropout >= 1.f) throw std::invalid_argument("Layer: dropout must be in [0, 1)");

	const size_t out = static_cast<size_t>(out_features);

	weights.init(out * in_features, true);
	bias.init(out, false);

	if (batchnorm)
	{
		gamma.init(out, false, 1.f);
		beta.init(out, false);
		running_mean.assign(out, 0.f);
		running_var.assign(out, 1.f);
	}
}

void Layer::init_parameters(std::mt19937& rng)
{
	const float gain = ReLU ? 2.f : 1.f;
	const float stddev = std::sqrt(gain / static_cast<float>(in_features));

	for (float& w : weights.w) w = stddev * random_normal(rng);

	std::fill(bias.w.begin(), bias.w.end(), 0.f);

	if (batchnorm)
	{
		std::fill(gamma.w.begin(), gamma.w.end(), 1.f);
		std::fill(beta.w.begin(), beta.w.end(), 0.f);
		std::fill(running_mean.begin(), running_mean.end(), 0.f);
		std::fill(running_var.begin(), running_var.end(), 1.f);
	}

	reset_optimizer_state();
}

const std::vector<float>& Layer::forward(const float* inputs, int batch, bool training, std::mt19937& rng)
{
	const int out = out_features;
	const size_t count = static_cast<size_t>(batch) * out;

	inputs_ = inputs;
	batch_ = batch;
	outputs.resize(count);

	kernels::linear_forward(inputs, weights.w.data(), batchnorm ? nullptr : bias.w.data(), outputs.data(), batch, in_features, out);

	if (batchnorm)
	{
		float* y = outputs.data();

		if (training)
		{
			std::vector<float> mean(out, 0.f);
			std::vector<float> var(out, 0.f);

			normalized_.resize(count);
			inv_std_.resize(out);

			for (int s = 0; s < batch; ++s)
			{
				const float* row = y + static_cast<size_t>(s) * out;

				for (int o = 0; o < out; ++o) mean[o] += row[o];
			}

			for (int o = 0; o < out; ++o) mean[o] /= static_cast<float>(batch);

			for (int s = 0; s < batch; ++s)
			{
				const float* row = y + static_cast<size_t>(s) * out;

				for (int o = 0; o < out; ++o)
				{
					const float d = row[o] - mean[o];
					var[o] += d * d;
				}
			}

			for (int o = 0; o < out; ++o)
			{
				var[o] /= static_cast<float>(batch);
				inv_std_[o] = 1.f / std::sqrt(var[o] + bn_epsilon);
			}

			for (int s = 0; s < batch; ++s)
			{
				float* row = y + static_cast<size_t>(s) * out;
				float* xhat = normalized_.data() + static_cast<size_t>(s) * out;

				for (int o = 0; o < out; ++o)
				{
					xhat[o] = (row[o] - mean[o]) * inv_std_[o];
					row[o] = gamma.w[o] * xhat[o] + beta.w[o];
				}
			}

			const float unbiased = batch > 1 ? static_cast<float>(batch) / static_cast<float>(batch - 1) : 1.f;

			for (int o = 0; o < out; ++o)
			{
				running_mean[o] = (1.f - bn_momentum) * running_mean[o] + bn_momentum * mean[o];
				running_var[o] = (1.f - bn_momentum) * running_var[o] + bn_momentum * var[o] * unbiased;
			}
		}
		else
		{
			std::vector<float> scale(out);
			std::vector<float> shift(out);

			for (int o = 0; o < out; ++o)
			{
				scale[o] = gamma.w[o] / std::sqrt(running_var[o] + bn_epsilon);
				shift[o] = beta.w[o] - running_mean[o] * scale[o];
			}

			for (int s = 0; s < batch; ++s)
			{
				float* row = y + static_cast<size_t>(s) * out;

				for (int o = 0; o < out; ++o) row[o] = row[o] * scale[o] + shift[o];
			}
		}
	}

	if (ReLU)
	{
		for (float& value : outputs) value = value < 0.f ? 0.f : value;
	}

	dropout_active_ = training && dropout > 0.f;

	if (dropout_active_)
	{
		const uint32_t threshold = static_cast<uint32_t>(dropout * 65536.f + 0.5f);
		const float keep_scale = 65536.f / static_cast<float>(65536u - threshold);

		const uint64_t high = rng();
		const uint64_t low = rng();
		uint64_t state = (high << 32) | low;

		dropout_mask_.resize(count);

		size_t i = 0;

		for (; i + 4 <= count; i += 4)
		{
			const uint64_t bits = mix64(state++);

			for (size_t j = 0; j < 4; ++j)
			{
				const uint32_t r = static_cast<uint32_t>((bits >> (16 * j)) & 0xFFFFu);

				dropout_mask_[i + j] = r >= threshold ? keep_scale : 0.f;
				outputs[i + j] *= dropout_mask_[i + j];
			}
		}

		if (i < count)
		{
			const uint64_t bits = mix64(state);

			for (size_t j = 0; i + j < count; ++j)
			{
				const uint32_t r = static_cast<uint32_t>((bits >> (16 * j)) & 0xFFFFu);

				dropout_mask_[i + j] = r >= threshold ? keep_scale : 0.f;
				outputs[i + j] *= dropout_mask_[i + j];
			}
		}
	}

	return outputs;
}

const float* Layer::backward(const float* grad_outputs, bool need_input_gradient)
{
	const int out = out_features;
	const int batch = batch_;
	const size_t count = static_cast<size_t>(batch) * out;

	scratch_.assign(grad_outputs, grad_outputs + count);
	float* g = scratch_.data();

	if (dropout_active_)
	{
		for (size_t i = 0; i < count; ++i) g[i] *= dropout_mask_[i];
	}

	if (ReLU)
	{
		for (size_t i = 0; i < count; ++i)
		{
			if (!(outputs[i] > 0.f)) g[i] = 0.f;
		}
	}

	if (batchnorm)
	{
		std::vector<float> sum_g(out, 0.f);
		std::vector<float> sum_gx(out, 0.f);

		for (int s = 0; s < batch; ++s)
		{
			const float* row = g + static_cast<size_t>(s) * out;
			const float* xhat = normalized_.data() + static_cast<size_t>(s) * out;

			for (int o = 0; o < out; ++o)
			{
				sum_g[o] += row[o];
				sum_gx[o] += row[o] * xhat[o];
			}
		}

		const float inv_batch = 1.f / static_cast<float>(batch);

		for (int o = 0; o < out; ++o)
		{
			beta.g[o] += sum_g[o];
			gamma.g[o] += sum_gx[o];
		}

		for (int s = 0; s < batch; ++s)
		{
			float* row = g + static_cast<size_t>(s) * out;
			const float* xhat = normalized_.data() + static_cast<size_t>(s) * out;

			for (int o = 0; o < out; ++o)
			{
				const float coefficient = gamma.w[o] * inv_std_[o] * inv_batch;
				row[o] = coefficient * (static_cast<float>(batch) * row[o] - sum_g[o] - xhat[o] * sum_gx[o]);
			}
		}
	}

	kernels::linear_backward_weights(inputs_, g, weights.g.data(), batchnorm ? nullptr : bias.g.data(), batch, in_features, out);

	if (!need_input_gradient) return nullptr;

	gradients.resize(static_cast<size_t>(batch) * in_features);
	kernels::linear_backward_input(g, weights.w.data(), gradients.data(), batch, in_features, out);

	return gradients.data();
}

void Layer::zero_gradients()
{
	weights.zero_grad();
	bias.zero_grad();
	gamma.zero_grad();
	beta.zero_grad();
}

void Layer::update(const OptimizerConfig& config, float learning_rate, long step)
{
	weights.step(config, learning_rate, step);

	if (batchnorm)
	{
		gamma.step(config, learning_rate, step);
		beta.step(config, learning_rate, step);
	}
	else bias.step(config, learning_rate, step);
}

void Layer::reset_optimizer_state()
{
	weights.reset_state();
	bias.reset_state();
	gamma.reset_state();
	beta.reset_state();
}

bool Layer::is_finite() const
{
	auto finite = [](const std::vector<float>& values)
	{
		return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
	};

	return finite(weights.w) && finite(bias.w) && finite(gamma.w) && finite(beta.w) && finite(running_mean) && finite(running_var);
}
