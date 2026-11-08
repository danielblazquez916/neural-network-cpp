#include "../augment.hpp"
#include "../drawing.hpp"
#include "../kernels.hpp"
#include "../neural_network.hpp"
#include "../utils.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{
	int failures = 0;
	int checks = 0;

#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #condition "\n"; } } while (0)
#define CHECK_NEAR(a, b, tolerance) do { ++checks; const double va = (a), vb = (b); if (!(std::fabs(va - vb) <= (tolerance) * (1.0 + std::fabs(vb)))) { ++failures; std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #a " = " << va << " vs " #b " = " << vb << "\n"; } } while (0)

	struct Quiet
	{
		std::ostringstream sink;
		std::streambuf* old;

		Quiet() : old(std::cout.rdbuf(sink.rdbuf())) {}
		~Quiet() { std::cout.rdbuf(old); }
	};

	struct Spec
	{
		int in;
		int out;
		bool relu;
		bool bn;
		float dropout;
	};

	std::vector<float> random_vector(std::mt19937& rng, size_t size, float stddev = 1.f)
	{
		std::normal_distribution<float> dist(0.f, stddev);
		std::vector<float> values(size);

		for (float& value : values) value = dist(rng);

		return values;
	}

	void make_network(neural_network& nn, const std::vector<Spec>& specs, unsigned seed)
	{
		nn.rng.seed(seed);
		nn.layers.clear();

		for (const Spec& spec : specs) nn.layers.emplace_back(spec.in, spec.out, spec.relu, spec.bn, spec.dropout);

		std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);

		for (Layer& layer : nn.layers)
		{
			layer.init_parameters(nn.rng);

			for (float& b : layer.bias.w) b = uniform(nn.rng);

			if (layer.batchnorm)
			{
				for (float& g : layer.gamma.w) g = 1.f + uniform(nn.rng);
				for (float& b : layer.beta.w) b = uniform(nn.rng);
			}
		}
	}

	void test_kernels(bool simd)
	{
		kernels::set_simd_enabled(simd);
		std::cout << "  " << (simd ? "AVX2 path" : "scalar path") << (simd && !kernels::cpu_has_avx2() ? " (no AVX2 on this CPU: runs the scalar code)" : "") << "\n";
		std::mt19937 rng(1);

		for (int B : { 1, 2, 3, 4, 5, 7, 8, 9, 13, 64 })
		{
			for (int I : { 1, 3, 7, 8, 9, 16, 17, 31, 33, 48, 784 })
			{
				for (int O : { 1, 2, 3, 4, 5, 6, 7, 10, 13 })
				{
					const auto X = random_vector(rng, static_cast<size_t>(B) * I);
					const auto W = random_vector(rng, static_cast<size_t>(O) * I);
					const auto bias = random_vector(rng, O);
					const auto dY = random_vector(rng, static_cast<size_t>(B) * O);

					std::vector<float> Y(static_cast<size_t>(B) * O);
					kernels::linear_forward(X.data(), W.data(), bias.data(), Y.data(), B, I, O);

					std::vector<float> dX(static_cast<size_t>(B) * I, 123.f);
					kernels::linear_backward_input(dY.data(), W.data(), dX.data(), B, I, O);

					auto dW = random_vector(rng, static_cast<size_t>(O) * I);
					auto db = random_vector(rng, O);
					const auto dW0 = dW;
					const auto db0 = db;
					kernels::linear_backward_weights(X.data(), dY.data(), dW.data(), db.data(), B, I, O);

					for (int s = 0; s < B; ++s)
					{
						for (int o = 0; o < O; ++o)
						{
							double expected = bias[o];

							for (int i = 0; i < I; ++i) expected += static_cast<double>(X[s * I + i]) * W[o * I + i];

							CHECK_NEAR(Y[s * O + o], expected, 1e-4);
						}

						for (int i = 0; i < I; ++i)
						{
							double expected = 0.0;

							for (int o = 0; o < O; ++o) expected += static_cast<double>(dY[s * O + o]) * W[o * I + i];

							CHECK_NEAR(dX[s * I + i], expected, 1e-4);
						}
					}

					for (int o = 0; o < O; ++o)
					{
						for (int i = 0; i < I; ++i)
						{
							double expected = dW0[o * I + i];

							for (int s = 0; s < B; ++s) expected += static_cast<double>(dY[s * O + o]) * X[s * I + i];

							CHECK_NEAR(dW[o * I + i], expected, 1e-4);
						}

						double expected = db0[o];

						for (int s = 0; s < B; ++s) expected += dY[s * O + o];

						CHECK_NEAR(db[o], expected, 1e-4);
					}
				}
			}
		}

		std::vector<float> X(6, 1.f), W(6, 1.f), Y(2), dW(6, 0.f);
		kernels::linear_forward(X.data(), W.data(), nullptr, Y.data(), 2, 3, 1);
		CHECK_NEAR(Y[0], 3.0, 1e-6);
		kernels::linear_backward_weights(X.data(), Y.data(), dW.data(), nullptr, 2, 3, 1);

		kernels::set_simd_enabled(true);
	}

	void test_adamw_kernel()
	{
		std::mt19937 rng(8);

		for (size_t n : { size_t(1), size_t(7), size_t(8), size_t(9), size_t(31), size_t(1000) })
		{
			const auto w0 = random_vector(rng, n);
			const auto g = random_vector(rng, n, 0.1f);
			auto m0 = random_vector(rng, n, 0.01f);
			auto v0 = random_vector(rng, n, 0.01f);

			for (float& v : v0) v = v * v;

			auto w1 = w0, m1 = m0, v1 = v0;
			auto w2 = w0, m2 = m0, v2 = v0;

			kernels::scalar::adamw_update(w1.data(), g.data(), m1.data(), v1.data(), n, 0.9f, 0.999f, 1e-8f, 0.01f, 3.f, 0.9999f);
			kernels::set_simd_enabled(true);
			kernels::adamw_update(w2.data(), g.data(), m2.data(), v2.data(), n, 0.9f, 0.999f, 1e-8f, 0.01f, 3.f, 0.9999f);

			for (size_t i = 0; i < n; ++i)
			{
				CHECK_NEAR(w1[i], w2[i], 1e-5);
				CHECK_NEAR(m1[i], m2[i], 1e-5);
				CHECK_NEAR(v1[i], v2[i], 1e-5);
			}
		}
	}

	void test_softmax_cross_entropy()
	{
		const float logits[] = { 0.f, 0.f };
		const uint8_t label0[] = { 0 };
		float probs[2], grad[2];
		int prediction = -1;

		double loss = softmax_cross_entropy(logits, 1, 2, label0, probs, grad, &prediction);
		CHECK_NEAR(loss, std::log(2.0), 1e-6);
		CHECK_NEAR(probs[0], 0.5, 1e-6);
		CHECK_NEAR(grad[0], -0.5, 1e-6);
		CHECK_NEAR(grad[1], 0.5, 1e-6);

		const float big[] = { 1000.f, 0.f };
		loss = softmax_cross_entropy(big, 1, 2, label0, probs, grad, &prediction);
		CHECK(std::isfinite(loss));
		CHECK_NEAR(loss, 0.0, 1e-6);
		CHECK(prediction == 0);

		const float wrong[] = { -1000.f, 1000.f };
		loss = softmax_cross_entropy(wrong, 1, 2, label0, probs, grad, &prediction);
		CHECK(std::isfinite(loss));
		CHECK_NEAR(loss, 2000.0, 1e-6);
		CHECK(prediction == 1);

		const float two[] = { 1.f, 2.f, 3.f, 0.f, 0.f, 0.f };
		const uint8_t labels[] = { 1, 0 };
		float grad2[6];
		loss = softmax_cross_entropy(two, 2, 3, labels, nullptr, grad2, nullptr);
		float column = 0.f;

		for (int c = 0; c < 3; ++c) column += grad2[c];

		CHECK_NEAR(column, 0.0, 1e-6);
		CHECK(std::isfinite(loss));

		const uint8_t invalid[] = { 5 };
		CHECK(std::isnan(softmax_cross_entropy(logits, 1, 2, invalid, nullptr, nullptr, nullptr)));
	}

	void test_optimizers()
	{
		OptimizerConfig sgd;
		sgd.type = OptimizerType::SGD;
		sgd.momentum = 0.9f;
		sgd.weight_decay = 0.f;

		Param p;
		p.init(1, true, 1.f);
		p.g[0] = 0.5f;
		p.step(sgd, 0.1f, 1);
		CHECK_NEAR(p.w[0], 0.95, 1e-6);
		p.step(sgd, 0.1f, 2);
		CHECK_NEAR(p.w[0], 0.855, 1e-6);

		sgd.weight_decay = 0.1f;
		Param q;
		q.init(2, true, 1.f);
		q.decay = false;
		q.g[0] = q.g[1] = 0.f;
		q.step(sgd, 0.1f, 1);
		CHECK_NEAR(q.w[0], 1.0, 1e-6);
		q.decay = true;
		q.step(sgd, 0.1f, 2);
		CHECK_NEAR(q.w[0], 0.99, 1e-6);

		OptimizerConfig adam;
		adam.weight_decay = 0.1f;
		Param a;
		a.init(1, true, 1.f);
		a.g[0] = 0.5f;
		a.step(adam, 0.1f, 1);
		CHECK_NEAR(a.w[0], 0.89, 1e-5);

		Param b;
		b.init(1, false, 1.f);
		b.g[0] = -3.f;
		b.step(adam, 0.01f, 1);
		CHECK_NEAR(b.w[0], 1.01, 1e-5);

		Param z;
		z.init(1, true, 1.f);
		z.step(adam, 0.01f, 1);
		CHECK(std::isfinite(z.w[0]));
	}

	void gradient_check(const char* name, const std::vector<Spec>& specs, bool use_dropout_seed)
	{
		const int batch = 7;
		const int classes = specs.back().out;
		neural_network nn(4, 1e-3f);
		make_network(nn, specs, 11);

		std::mt19937 rng(3);
		const auto x = random_vector(rng, static_cast<size_t>(batch) * specs.front().in);
		std::vector<uint8_t> labels(batch);

		for (int s = 0; s < batch; ++s) labels[s] = static_cast<uint8_t>(s % classes);

		std::vector<float> grad(static_cast<size_t>(batch) * classes);

		auto loss_of = [&](bool backprop)
		{
			if (use_dropout_seed) nn.rng.seed(777);

			const std::vector<float>& logits = nn.forward(x.data(), batch, true);
			const double loss = softmax_cross_entropy(logits.data(), batch, classes, labels.data(), nullptr, grad.data(), nullptr) / batch;

			if (backprop) nn.backward(grad.data());

			return loss;
		};

		auto activation_pattern = [&]()
		{
			std::vector<char> pattern;

			for (const Layer& layer : nn.layers)
			{
				if (layer.ReLU)
				{
					for (float value : layer.outputs) pattern.push_back(value > 0.f ? 1 : 0);
				}
			}

			return pattern;
		};

		loss_of(true);
		const std::vector<char> base_pattern = activation_pattern();

		std::vector<std::pair<std::vector<float>*, std::vector<float>*>> tensors;

		for (Layer& layer : nn.layers)
		{
			tensors.push_back({ &layer.weights.w, &layer.weights.g });

			if (layer.batchnorm)
			{
				tensors.push_back({ &layer.gamma.w, &layer.gamma.g });
				tensors.push_back({ &layer.beta.w, &layer.beta.g });
			}
			else tensors.push_back({ &layer.bias.w, &layer.bias.g });
		}

		std::vector<std::vector<float>> analytic;

		for (auto& tensor : tensors) analytic.push_back(*tensor.second);

		const float epsilon = 1e-3f;
		double worst = 0.0;
		int compared = 0;
		int skipped = 0;

		for (size_t t = 0; t < tensors.size(); ++t)
		{
			std::vector<float>& values = *tensors[t].first;

			for (size_t i = 0; i < values.size(); ++i)
			{
				const float original = values[i];

				values[i] = original + epsilon;
				const double plus = loss_of(false);
				const bool plus_same = activation_pattern() == base_pattern;
				values[i] = original - epsilon;
				const double minus = loss_of(false);
				const bool minus_same = activation_pattern() == base_pattern;
				values[i] = original;

				if (!plus_same || !minus_same)
				{
					++skipped;
					continue;
				}

				const double numeric = (plus - minus) / (2.0 * epsilon);
				const double error = std::fabs(numeric - analytic[t][i]);
				const double scale = std::max(std::fabs(numeric), std::fabs(static_cast<double>(analytic[t][i])));
				const double relative = error / (scale + 1e-2);

				worst = std::max(worst, relative);
				++compared;
				++checks;

				if (error > 2e-3 + 0.03 * scale)
				{
					++failures;
					std::cerr << "FAIL gradient " << name << " tensor " << t << " index " << i << ": analytic " << analytic[t][i] << " numeric " << numeric << "\n";
				}
			}
		}

		CHECK(skipped * 4 <= compared + skipped);
		CHECK(compared > 0);

		std::cout << "  gradient check '" << name << "': " << compared << " parameters, " << skipped << " skipped at a ReLU kink, worst relative error " << worst << "\n";
	}

	void test_gradients()
	{
		gradient_check("linear + relu + linear", { { 6, 5, true, false, 0.f }, { 5, 3, false, false, 0.f } }, false);
		gradient_check("batchnorm + relu stack", { { 6, 5, true, true, 0.f }, { 5, 4, true, true, 0.f }, { 4, 3, false, false, 0.f } }, false);
		gradient_check("dropout", { { 6, 8, true, false, 0.3f }, { 8, 3, false, false, 0.f } }, true);
		gradient_check("batchnorm + dropout", { { 6, 8, true, true, 0.25f }, { 8, 5, true, true, 0.f }, { 5, 3, false, false, 0.f } }, true);
		gradient_check("batchnorm without relu", { { 6, 4, false, true, 0.f }, { 4, 3, false, false, 0.f } }, false);
	}

	void test_dropout_and_batchnorm()
	{
		std::mt19937 rng(5);

		Layer layer(50, 400, false, false, 0.5f);
		layer.init_parameters(rng);

		Layer reference = layer;
		reference.dropout = 0.f;

		const int batch = 20;
		const auto x = random_vector(rng, static_cast<size_t>(batch) * 50);

		const auto& dropped = layer.forward(x.data(), batch, true, rng);
		const auto& clean = reference.forward(x.data(), batch, true, rng);

		size_t zeros = 0;

		for (size_t i = 0; i < dropped.size(); ++i)
		{
			if (dropped[i] == 0.f) ++zeros;
			else CHECK_NEAR(dropped[i], 2.0 * clean[i], 1e-5);
		}

		const double fraction = static_cast<double>(zeros) / static_cast<double>(dropped.size());
		CHECK(fraction > 0.46 && fraction < 0.54);

		const auto& inference = layer.forward(x.data(), batch, false, rng);
		const std::vector<float> inference_copy = inference;
		const auto& clean_inference = reference.forward(x.data(), batch, false, rng);

		for (size_t i = 0; i < inference_copy.size(); ++i) CHECK_NEAR(inference_copy[i], clean_inference[i], 1e-6);

		Layer bn(30, 6, false, true, 0.f);
		bn.init_parameters(rng);

		const int bn_batch = 64;
		auto input = random_vector(rng, static_cast<size_t>(bn_batch) * 30, 3.f);

		for (float& value : input) value += 5.f;

		const std::vector<float> out = bn.forward(input.data(), bn_batch, true, rng);

		for (int o = 0; o < 6; ++o)
		{
			double mean = 0.0, var = 0.0;

			for (int s = 0; s < bn_batch; ++s) mean += out[s * 6 + o];

			mean /= bn_batch;

			for (int s = 0; s < bn_batch; ++s) var += (out[s * 6 + o] - mean) * (out[s * 6 + o] - mean);

			var /= bn_batch;

			CHECK(std::fabs(mean) < 1e-4);
			CHECK(std::fabs(var - 1.0) < 1e-3);
		}

		for (int i = 0; i < 200; ++i) bn.forward(input.data(), bn_batch, true, rng);

		const std::vector<float> training_out = bn.forward(input.data(), bn_batch, true, rng);
		const std::vector<float> eval_out = bn.forward(input.data(), bn_batch, false, rng);

		for (size_t i = 0; i < eval_out.size(); ++i) CHECK(std::fabs(eval_out[i] - training_out[i]) < 0.02f * (1.f + std::fabs(training_out[i])));

		Layer single(30, 6, true, true, 0.f);
		single.init_parameters(rng);
		const std::vector<float> one = single.forward(input.data(), 1, true, rng);
		CHECK(std::all_of(one.begin(), one.end(), [](float v) { return std::isfinite(v); }));
		CHECK(single.is_finite());
	}

	std::vector<float> eval_logits(neural_network& nn, const std::vector<float>& x, int batch)
	{
		return nn.forward(x.data(), batch, false);
	}

	void test_serialization()
	{
		const std::string path = "test_model_tmp.json";
		std::filesystem::remove(path);

		neural_network nn(8, 1e-3f);
		nn.rng.seed(21);
		nn.build_default_layers();
		nn.epochs_trained = 7;

		std::mt19937 rng(9);
		const auto batch_input = random_vector(rng, static_cast<size_t>(16) * neural_network::input_size);

		for (int i = 0; i < 5; ++i) nn.forward(batch_input.data(), 16, true);

		{
			Quiet quiet;
			CHECK(nn.serialize(path.c_str()));
		}

		CHECK(std::filesystem::exists(path));
		CHECK(!std::filesystem::exists(path + ".tmp"));

		neural_network loaded(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(loaded.deserialize(path.c_str()));
		}

		CHECK(loaded.layers.size() == nn.layers.size());
		CHECK(loaded.epochs_trained == 7);
		CHECK(loaded.input_mean == nn.input_mean);
		CHECK(loaded.input_std == nn.input_std);

		const auto expected = eval_logits(nn, batch_input, 16);
		const auto actual = eval_logits(loaded, batch_input, 16);

		CHECK(expected.size() == actual.size());

		bool identical = expected.size() == actual.size();

		for (size_t i = 0; identical && i < expected.size(); ++i) identical = expected[i] == actual[i];

		CHECK(identical);

		const auto before = std::filesystem::file_size(path);
		nn.layers[1].weights.w[3] = std::nanf("");

		{
			Quiet quiet;
			CHECK(!nn.serialize(path.c_str()));
		}

		CHECK(std::filesystem::file_size(path) == before);
		CHECK(!std::filesystem::exists(path + ".tmp"));

		std::filesystem::remove(path);
	}

	void write_text(const std::string& path, const std::string& text)
	{
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		file << text;
	}

	void test_legacy_and_invalid_files()
	{
		const std::string path = "test_legacy_tmp.json";
		std::mt19937 rng(77);

		const int hidden = 5;
		const auto w0 = random_vector(rng, static_cast<size_t>(hidden) * neural_network::input_size, 0.05f);
		const auto b0 = random_vector(rng, hidden, 0.1f);
		const auto w1 = random_vector(rng, static_cast<size_t>(10) * hidden, 0.5f);
		const auto b1 = random_vector(rng, 10, 0.1f);

		nlohmann::json legacy;

		for (int j = 0; j < hidden; ++j)
		{
			legacy["layers"][0]["neurons"][j]["weights"] = std::vector<float>(w0.begin() + j * neural_network::input_size, w0.begin() + (j + 1) * neural_network::input_size);
			legacy["layers"][0]["neurons"][j]["bias"] = b0[j];
		}

		for (int j = 0; j < 10; ++j)
		{
			legacy["layers"][1]["neurons"][j]["weights"] = std::vector<float>(w1.begin() + j * hidden, w1.begin() + (j + 1) * hidden);
			legacy["layers"][1]["neurons"][j]["bias"] = b1[j];
		}

		write_text(path, legacy.dump(4));

		neural_network nn(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(nn.deserialize(path.c_str()));
		}

		CHECK(nn.layers.size() == 2);
		CHECK(nn.input_mean == 0.f && nn.input_std == 1.f);
		CHECK(nn.layers[0].ReLU && !nn.layers[1].ReLU);

		const auto x = random_vector(rng, neural_network::input_size, 0.5f);
		const auto logits = eval_logits(nn, x, 1);

		std::vector<double> h(hidden);

		for (int j = 0; j < hidden; ++j)
		{
			double sum = b0[j];

			for (int i = 0; i < neural_network::input_size; ++i) sum += static_cast<double>(w0[j * neural_network::input_size + i]) * x[i];

			h[j] = sum > 0.0 ? sum : 0.0;
		}

		for (int o = 0; o < 10; ++o)
		{
			double sum = b1[o];

			for (int j = 0; j < hidden; ++j) sum += static_cast<double>(w1[o * hidden + j]) * h[j];

			CHECK_NEAR(logits[o], sum, 1e-4);
		}

		const std::vector<std::pair<std::string, std::string>> broken =
		{
			{ "garbage", "this is not json {{{" },
			{ "no layers", "{\"version\":2,\"layers\":[]}" },
			{ "wrong input size", "{\"version\":2,\"layers\":[{\"in\":3,\"out\":10,\"relu\":false,\"batchnorm\":false,\"dropout\":0,\"weights\":[],\"bias\":[]}]}" },
			{ "future version", "{\"version\":99,\"layers\":[]}" },
			{ "truncated", legacy.dump().substr(0, 400) },
		};

		for (const auto& [name, text] : broken)
		{
			write_text(path, text);
			neural_network bad(8, 1e-3f);
			Quiet quiet;
			const bool ok = bad.deserialize(path.c_str());
			CHECK(!ok);
			CHECK(bad.layers.empty());
		}

		std::filesystem::remove(path);
		neural_network fresh(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(fresh.deserialize(path.c_str()));
		}

		CHECK(fresh.layers.size() == 4);
		CHECK(fresh.layers.front().in_features == neural_network::input_size);
		CHECK(fresh.layers.back().out_features == neural_network::num_classes);

		write_text(path, "");
		neural_network empty(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(empty.deserialize(path.c_str()));
		}

		CHECK(empty.layers.size() == 4);

		std::filesystem::remove(path);
	}

	void test_invalid_arguments_and_divergence()
	{
		bool threw = false;

		try { neural_network bad(1, 1e-3f); }
		catch (const std::invalid_argument&) { threw = true; }

		CHECK(threw);

		threw = false;

		try { neural_network bad(32, 0.f); }
		catch (const std::invalid_argument&) { threw = true; }

		CHECK(threw);

		threw = false;

		try { Layer bad(4, 4, true, false, 1.f); }
		catch (const std::invalid_argument&) { threw = true; }

		CHECK(threw);

		neural_network nn(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(nn.process(false, false).samples == 0);
			nn.build_default_layers();
			CHECK(nn.process(false, true).samples == 0);
		}

		nn.images.push_back({ std::vector<float>(neural_network::input_size, 0.5f), 3 });
		nn.images.push_back({ std::vector<float>(neural_network::input_size, 0.2f), 4 });
		nn.layers[0].weights.w[0] = std::nanf("");

		{
			Quiet quiet;
			CHECK(nn.process(false, false).diverged);
		}
	}

	void test_random_helpers()
	{
		std::mt19937 rng(3);

		std::vector<int> seen(7, 0);

		for (int i = 0; i < 7000; ++i)
		{
			const uint32_t value = random_below(rng, 7);
			CHECK(value < 7);
			++seen[value];
		}

		for (int count : seen) CHECK(count > 800 && count < 1200);

		CHECK(random_below(rng, 1) == 0);

		double sum = 0.0, square = 0.0;
		const int n = 20000;

		for (int i = 0; i < n; ++i)
		{
			const double u = random_uniform(rng);
			CHECK(u >= 0.0 && u < 1.0);
			sum += u;
		}

		CHECK(std::fabs(sum / n - 0.5) < 0.01);

		sum = 0.0;

		for (int i = 0; i < n; ++i)
		{
			const double z = random_normal(rng);
			CHECK(std::isfinite(z));
			sum += z;
			square += z * z;
		}

		CHECK(std::fabs(sum / n) < 0.03);
		CHECK(std::fabs(square / n - 1.0) < 0.05);

		std::vector<size_t> a(500), b(500);

		for (size_t i = 0; i < a.size(); ++i) a[i] = b[i] = i;

		std::mt19937 r1(11), r2(11);
		shuffle_indices(a, r1);
		shuffle_indices(b, r2);
		CHECK(a == b);

		std::vector<size_t> sorted = a;
		std::sort(sorted.begin(), sorted.end());

		for (size_t i = 0; i < sorted.size(); ++i) CHECK(sorted[i] == i);

		CHECK(a[0] != 0 || a[1] != 1);
	}

	void test_label_smoothing()
	{
		const float logits[] = { 1.f, 2.f, 0.5f, -1.f };
		const uint8_t label[] = { 1 };
		float grad_plain[4], grad_smooth[4], probs[4];

		const double loss_plain = softmax_cross_entropy(logits, 1, 4, label, probs, grad_plain, nullptr, 0.f);
		const double loss_smooth = softmax_cross_entropy(logits, 1, 4, label, probs, grad_smooth, nullptr, 0.1f);

		CHECK_NEAR(loss_plain, loss_smooth, 1e-12);

		float sum = 0.f;

		for (int c = 0; c < 4; ++c) sum += grad_smooth[c];

		CHECK_NEAR(sum, 0.0, 1e-6);

		CHECK_NEAR(grad_smooth[1], probs[1] - 0.925, 1e-6);
		CHECK_NEAR(grad_smooth[0], probs[0] - 0.025, 1e-6);
		CHECK(std::fabs(grad_plain[1] - grad_smooth[1]) > 0.01);
	}

	void test_augmentation()
	{
		std::mt19937 rng(2);
		std::vector<float> digit(784, 0.f);

		for (int y = 6; y < 22; ++y)
		{
			for (int x = 12; x < 16; ++x) digit[y * 28 + x] = 1.f;
		}

		AugmentConfig none;
		none.enabled = true;
		none.max_shift = none.max_rotation = none.max_scale = none.max_shear = 0.f;
		none.elastic_alpha = 0.f;

		Augmenter identity(none, 1);
		std::vector<float> out(784);
		identity.apply(digit.data(), out.data(), 77);

		for (int i = 0; i < 784; ++i) CHECK_NEAR(out[i], digit[i], 1e-6);

		AugmentConfig config;
		config.enabled = true;

		Augmenter warp(config, 5);
		std::vector<float> first(784), again(784), other(784);
		warp.apply(digit.data(), first.data(), 1234);
		warp.apply(digit.data(), again.data(), 1234);
		warp.apply(digit.data(), other.data(), 4321);

		CHECK(first == again);
		CHECK(first != other);

		for (float value : first) CHECK(std::isfinite(value) && value >= -1e-6f && value <= 1.0001f);

		double mass = 0.0;

		for (float value : first) mass += value;

		CHECK(mass > 30.0 && mass < 110.0);

		Augmenter same_pool(config, 5), other_pool(config, 6);
		std::vector<float> a(784), b(784), c(784);
		warp.apply(digit.data(), a.data(), 99);
		same_pool.apply(digit.data(), b.data(), 99);
		other_pool.apply(digit.data(), c.data(), 99);
		CHECK(a == b);
		CHECK(a != c);

		AugmentConfig wild;
		wild.enabled = true;
		wild.max_shift = 40.f;
		wild.max_rotation = 180.f;
		wild.max_scale = 0.9f;
		wild.max_shear = 2.f;
		wild.elastic_alpha = 200.f;

		Augmenter chaos(wild, 3);

		for (uint64_t seed = 0; seed < 200; ++seed)
		{
			chaos.apply(digit.data(), out.data(), seed);

			for (float value : out) CHECK(std::isfinite(value) && value >= -1e-6f && value <= 1.0001f);
		}
	}

	void test_drawing_preprocessing()
	{
		std::vector<float> out;

		CHECK(!preprocess_drawing(std::vector<float>(40 * 40, 0.f), 40, 40, out));

		std::vector<float> dot(40 * 40, 0.f);
		dot[20 * 40 + 20] = 1.f;
		CHECK(!preprocess_drawing(dot, 40, 40, out));
		CHECK(!preprocess_drawing(dot, 40, 39, out));
		CHECK(out.empty());

		auto seven = [](int size, int top, int left)
		{
			std::vector<float> grid(40 * 40, 0.f);

			for (int i = 0; i < size; ++i) grid[top * 40 + left + i] = 1.f;
			for (int i = 0; i < size; ++i) grid[(top + i) * 40 + left + size - 1 - i / 2] = 1.f;

			return grid;
		};

		std::vector<float> centered, moved, big;
		CHECK(preprocess_drawing(seven(14, 5, 5), 40, 40, centered));
		CHECK(preprocess_drawing(seven(14, 20, 22), 40, 40, moved));
		CHECK(preprocess_drawing(seven(30, 4, 4), 40, 40, big));

		CHECK(centered.size() == 784);
		CHECK(centered == moved);

		double difference = 0.0, mass = 0.0;

		for (int i = 0; i < 784; ++i)
		{
			difference += std::fabs(centered[i] - big[i]);
			mass += centered[i];
		}

		CHECK(difference < 0.35 * mass);

		double sum = 0.0, cy = 0.0, cx = 0.0;
		float peak = 0.f;

		for (int y = 0; y < 28; ++y)
		{
			for (int x = 0; x < 28; ++x)
			{
				const float value = centered[y * 28 + x];
				CHECK(value >= 0.f && value <= 1.f);
				sum += value;
				cy += value * y;
				cx += value * x;
				peak = std::max(peak, value);
			}
		}

		CHECK(std::fabs(cy / sum - 14.0) < 0.6);
		CHECK(std::fabs(cx / sum - 14.0) < 0.6);
		CHECK(peak > 0.9f);

		std::vector<float> plus(40 * 40, 0.f), cross;

		for (int i = 8; i < 33; ++i)
		{
			plus[20 * 40 + i] = 1.f;
			plus[i * 40 + 20] = 1.f;
		}

		CHECK(preprocess_drawing(plus, 40, 40, cross));

		for (int i = 0; i < 28; ++i)
		{
			CHECK(cross[i] == 0.f && cross[27 * 28 + i] == 0.f && cross[i * 28] == 0.f && cross[i * 28 + 27] == 0.f);
		}

		std::vector<float> line(40 * 40, 0.f), one;

		for (int y = 3; y < 37; ++y) line[y * 40 + 30] = 1.f;

		CHECK(preprocess_drawing(line, 40, 40, one));

		int columns = 0;

		for (int x = 0; x < 28; ++x)
		{
			float column_peak = 0.f;

			for (int y = 0; y < 28; ++y) column_peak = std::max(column_peak, one[y * 28 + x]);

			if (column_peak > 0.05f) ++columns;
		}

		CHECK(columns >= 2 && columns <= 6);

		const std::string art = ascii_digit(centered);
		CHECK(std::count(art.begin(), art.end(), '\n') == 28);
		CHECK(ascii_digit(std::vector<float>(5, 0.f)).empty());
	}

	void test_validation_split()
	{
		neural_network nn(8, 1e-3f);

		for (int i = 0; i < 60000; ++i) nn.images.push_back({ std::vector<float>(1, static_cast<float>(i)), static_cast<uint8_t>(i % 10) });

		nn.hold_out_validation(12);

		CHECK(nn.images.size() + nn.validation.size() == 60000);
		CHECK(nn.validation.size() > 4700 && nn.validation.size() < 5300);

		std::vector<char> in_validation(60000, 0);

		for (const auto& sample : nn.validation) in_validation[static_cast<size_t>(sample.first[0])] = 1;
		for (const auto& sample : nn.images) CHECK(in_validation[static_cast<size_t>(sample.first[0])] == 0);

		std::vector<int> per_residue(6, 0);

		for (const auto& sample : nn.validation) ++per_residue[static_cast<size_t>(sample.first[0]) % 6];

		for (int count : per_residue) CHECK(count > 0.8 * nn.validation.size() / 6 && count < 1.2 * nn.validation.size() / 6);

		neural_network again(8, 1e-3f);

		for (int i = 0; i < 60000; ++i) again.images.push_back({ std::vector<float>(1, static_cast<float>(i)), static_cast<uint8_t>(i % 10) });

		again.hold_out_validation(12);
		CHECK(again.validation.size() == nn.validation.size());
		CHECK(again.validation.front().first == nn.validation.front().first);
		CHECK(again.validation.back().first == nn.validation.back().first);
	}

	void test_learning_rate_schedule()
	{
		neural_network nn(8, 1e-2f);
		nn.rng.seed(1);
		nn.build_default_layers();

		CHECK_NEAR(nn.current_learning_rate(), 1e-2, 1e-6);
		nn.epochs_trained = 5;
		CHECK_NEAR(nn.current_learning_rate(), 1e-2 * std::pow(0.85, 5), 1e-6);
		nn.epochs_trained = 500;
		CHECK_NEAR(nn.current_learning_rate(), 1e-4, 1e-6);

		nn.epochs_trained = 3;
		nn.begin_training_session(10);
		CHECK_NEAR(nn.current_learning_rate(), 1e-2, 1e-6);
		nn.epochs_trained = 8;
		CHECK_NEAR(nn.current_learning_rate(), 0.5e-2, 1e-6);
		nn.epochs_trained = 13;
		CHECK_NEAR(nn.current_learning_rate(), 0.0, 1e-6);
		nn.epochs_trained = 30;
		CHECK_NEAR(nn.current_learning_rate(), 0.0, 1e-6);

		float previous = 1.f;

		for (int epoch = 3; epoch <= 13; ++epoch)
		{
			nn.epochs_trained = epoch;
			CHECK(nn.current_learning_rate() <= previous + 1e-9f);
			previous = nn.current_learning_rate();
		}

		nn.begin_training_session(0);
		nn.epochs_trained = 0;
		CHECK_NEAR(nn.current_learning_rate(), 1e-2, 1e-6);
	}

	void test_training_sanity()
	{
		neural_network nn(16, 1e-3f);
		nn.rng.seed(8);
		nn.build_default_layers();

		std::mt19937 rng(4);
		const int batch = 64;
		const auto x = random_vector(rng, static_cast<size_t>(batch) * neural_network::input_size);
		std::vector<uint8_t> labels(batch);

		for (int s = 0; s < batch; ++s) labels[s] = static_cast<uint8_t>(s % 10);

		const auto& logits = nn.forward(x.data(), batch, false);
		const double initial = softmax_cross_entropy(logits.data(), batch, 10, labels.data(), nullptr, nullptr, nullptr) / batch;

		CHECK(initial > 1.5 && initial < 3.5);

		nn.warmup_steps = 0;
		std::vector<float> grad(static_cast<size_t>(batch) * 10);
		double last = initial;

		for (int step = 0; step < 150; ++step)
		{
			const auto& out = nn.forward(x.data(), batch, true);
			last = softmax_cross_entropy(out.data(), batch, 10, labels.data(), nullptr, grad.data(), nullptr) / batch;
			nn.backward(grad.data());
			nn.apply_gradients();
		}

		CHECK(last < 0.05);

		const auto together = nn.forward(x.data(), batch, false);

		for (int s : { 0, 17, 63 })
		{
			const auto alone = nn.forward(x.data() + static_cast<size_t>(s) * neural_network::input_size, 1, false);

			for (int c = 0; c < 10; ++c) CHECK_NEAR(alone[c], together[static_cast<size_t>(s) * 10 + c], 1e-3);
		}
	}

	void test_missing_model_and_holdout_roundtrip()
	{
		const std::string path = "test_missing_tmp.json";
		std::filesystem::remove(path);

		neural_network nn(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(!nn.deserialize(path.c_str(), false));
			CHECK(nn.layers.empty());
			CHECK(nn.deserialize(path.c_str(), true));
		}

		CHECK(nn.layers.size() == 4);
		CHECK(nn.holdout_every == 0);

		nn.holdout_every = 12;

		{
			Quiet quiet;
			CHECK(nn.serialize(path.c_str()));
		}

		neural_network loaded(8, 1e-3f);

		{
			Quiet quiet;
			CHECK(loaded.deserialize(path.c_str(), false));
		}

		CHECK(loaded.holdout_every == 12);
		std::filesystem::remove(path);
	}

	void test_threads_do_not_change_results()
	{
#if defined(_OPENMP)
		std::mt19937 rng(6);
		const int B = 64, I = 784, O = 256;
		const auto X = random_vector(rng, static_cast<size_t>(B) * I);
		const auto W = random_vector(rng, static_cast<size_t>(O) * I, 0.1f);
		const auto bias = random_vector(rng, O);
		const auto dY = random_vector(rng, static_cast<size_t>(B) * O);
		const auto g = random_vector(rng, 300000, 0.1f);

		auto run = [&](int threads)
		{
			omp_set_num_threads(threads);
			std::vector<std::vector<float>> results;
			std::vector<float> Y(static_cast<size_t>(B) * O), dX(static_cast<size_t>(B) * I), dW(static_cast<size_t>(O) * I, 0.f), db(O, 0.f);
			kernels::linear_forward(X.data(), W.data(), bias.data(), Y.data(), B, I, O);
			kernels::linear_backward_input(dY.data(), W.data(), dX.data(), B, I, O);
			kernels::linear_backward_weights(X.data(), dY.data(), dW.data(), db.data(), B, I, O);
			auto w = random_vector(rng = std::mt19937(1), g.size()), m = w, v = w;

			for (float& value : v) value = value * value;

			kernels::adamw_update(w.data(), g.data(), m.data(), v.data(), g.size(), 0.9f, 0.999f, 1e-8f, 0.01f, 3.f, 0.999f);
			results.push_back(Y); results.push_back(dX); results.push_back(dW); results.push_back(db); results.push_back(w);

			return results;
		};

		const auto one = run(1);
		const auto four = run(4);
		const auto three = run(3);

		CHECK(one == four);
		CHECK(one == three);
		std::cout << "  " << omp_get_num_procs() << " processors: results identical with 1, 3 and 4 threads\n";
#else
		std::cout << "  (built without OpenMP, skipping)\n";
#endif
	}

	void test_drawing_flow()
	{
		neural_network nn(8, 1e-3f);
		nn.rng.seed(4);
		nn.build_default_layers();

		std::vector<float> drawing(neural_network::input_size, 0.f);

		for (int i = 0; i < 28; ++i) drawing[14 * 28 + i] = 1.f;

		nn.images.push_back({ drawing, 7 });

		std::ostringstream capture;
		std::streambuf* old = std::cout.rdbuf(capture.rdbuf());
		const EpochStats stats = nn.process(true, false);
		std::cout.rdbuf(old);

		const std::string text = capture.str();

		CHECK(stats.samples == 1);
		CHECK(text.find("Prediction:") != std::string::npos);
		CHECK(text.find("Real Number: 7") != std::string::npos);
		CHECK(text.find("%") != std::string::npos);
	}

	void test_training_on_mnist()
	{
		if (!std::filesystem::exists("dataset/images/train-images.idx3-ubyte"))
		{
			std::cout << "  (dataset not found, skipping the MNIST test)\n";
			return;
		}

		neural_network nn(32, 1e-3f);
		nn.rng.seed(2024);
		nn.build_default_layers();

		{
			Quiet quiet;
			CHECK(nn.read_images("train"));
		}

		CHECK(nn.images.size() == 60000);

		std::vector<std::pair<std::vector<float>, uint8_t>> train(nn.images.begin(), nn.images.begin() + 6000);
		nn.images.clear();

		{
			Quiet quiet;
			CHECK(nn.read_images("t10k"));
		}

		CHECK(nn.images.size() == 10000);

		std::vector<std::pair<std::vector<float>, uint8_t>> test(nn.images.begin(), nn.images.begin() + 2000);

		nn.images = train;

		double first_loss = 0.0, last_loss = 0.0;

		for (int epoch = 0; epoch < 3; ++epoch)
		{
			Quiet quiet;
			const EpochStats stats = nn.process(false, true);

			CHECK(!stats.diverged);
			CHECK(stats.samples == 6000);

			if (epoch == 0) first_loss = stats.loss;

			last_loss = stats.loss;
		}

		CHECK(nn.epochs_trained == 3);
		CHECK(last_loss < first_loss);

		nn.images = test;
		EpochStats result;

		{
			Quiet quiet;
			result = nn.process(false, false);
		}

		std::cout << "  MNIST subset (6000 train images, 3 epochs): test accuracy on 2000 images = " << result.accuracy * 100.0 << "%\n";
		CHECK(result.accuracy > 0.90);

		auto run = [&](unsigned seed)
		{
			neural_network a(32, 1e-3f);
			a.rng.seed(seed);
			a.build_default_layers();
			a.images.assign(train.begin(), train.begin() + 500);
			Quiet quiet;
			return a.process(false, true).loss;
		};

		CHECK(run(5) == run(5));
		CHECK(run(5) != run(6));
	}
}

int main()
{
	std::cout << "kernels\n";
	test_kernels(false);
	test_kernels(true);
	std::cout << "adamw kernel\n";
	test_adamw_kernel();
	std::cout << "softmax / cross-entropy\n";
	test_softmax_cross_entropy();
	std::cout << "optimizers\n";
	test_optimizers();
	std::cout << "gradients\n";
	test_gradients();
	std::cout << "dropout / batchnorm\n";
	test_dropout_and_batchnorm();
	std::cout << "serialization\n";
	test_serialization();
	std::cout << "legacy format and invalid files\n";
	test_legacy_and_invalid_files();
	std::cout << "random helpers\n";
	test_random_helpers();
	std::cout << "label smoothing\n";
	test_label_smoothing();
	std::cout << "augmentation\n";
	test_augmentation();
	std::cout << "drawing preprocessing\n";
	test_drawing_preprocessing();
	std::cout << "validation split\n";
	test_validation_split();
	std::cout << "learning rate schedule\n";
	test_learning_rate_schedule();
	std::cout << "training sanity\n";
	test_training_sanity();
	std::cout << "missing model and hold-out roundtrip\n";
	test_missing_model_and_holdout_roundtrip();
	std::cout << "threads\n";
	test_threads_do_not_change_results();
	std::cout << "arguments and divergence\n";
	test_invalid_arguments_and_divergence();
	std::cout << "drawing flow\n";
	test_drawing_flow();
	std::cout << "training on MNIST\n";
	test_training_on_mnist();

	std::cout << "\n" << checks << " checks, " << failures << " failures\n";

	return failures == 0 ? 0 : 1;
}
