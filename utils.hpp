#pragma once

#include <vector>
#include <fstream>

std::pair<float, float> get_biggest(std::vector<float>& nums);

uint32_t read_uint32(std::ifstream& file);