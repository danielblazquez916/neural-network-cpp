#include "optimizer.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <cmath>

void Param::init(size_t size, bool apply_decay, float value)
{
	w.assign(size, value);
	g.assign(size, 0.f);
	m.assign(size, 0.f);
	v.assign(size, 0.f);
	decay = apply_decay;
}

void Param::zero_grad()
{
	std::fill(g.begin(), g.end(), 0.f);
}

void Param::reset_state()
{
	std::fill(m.begin(), m.end(), 0.f);
	std::fill(v.begin(), v.end(), 0.f);
}

void Param::step(const OptimizerConfig& config, float learning_rate, long step)
{
	const size_t size = w.size();

	if (config.type == OptimizerType::SGD)
	{
		for (size_t i = 0; i < size; ++i)
		{
			float grad = g[i];

			if (decay) grad += config.weight_decay * w[i];

			m[i] = config.momentum * m[i] + grad;
			w[i] -= learning_rate * m[i];
		}

		return;
	}

	const double t = static_cast<double>(step < 1 ? 1 : step);
	const float bias_correction1 = static_cast<float>(1.0 - std::pow(static_cast<double>(config.beta1), t));
	const float inv_sqrt_bias_correction2 = static_cast<float>(1.0 / std::sqrt(1.0 - std::pow(static_cast<double>(config.beta2), t)));
	const float step_size = learning_rate / bias_correction1;
	const float shrink = decay ? 1.f - learning_rate * config.weight_decay : 1.f;

	kernels::adamw_update(w.data(), g.data(), m.data(), v.data(), size, config.beta1, config.beta2, config.epsilon,
		step_size, inv_sqrt_bias_correction2, shrink);
}
