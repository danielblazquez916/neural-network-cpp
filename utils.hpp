#pragma once

#include <cstdint>
#include <fstream>
#include <random>
#include <utility>
#include <vector>

std::pair<float, int> get_biggest(const std::vector<float>& nums);

uint32_t read_uint32(std::ifstream& file);

double softmax_cross_entropy(const float* logits, int batch, int classes, const uint8_t* labels,
	float* probs, float* grad, int* predictions, float label_smoothing = 0.f);

uint32_t random_below(std::mt19937& rng, uint32_t n);
float random_uniform(std::mt19937& rng);
float random_normal(std::mt19937& rng);
void shuffle_indices(std::vector<size_t>& indices, std::mt19937& rng);

inline uint64_t mix64(uint64_t value)
{
	value += 0x9E3779B97F4A7C15ull;
	value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
	value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;

	return value ^ (value >> 31);
}
