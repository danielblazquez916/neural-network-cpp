#pragma once

#include <string>
#include <vector>

bool preprocess_drawing(const std::vector<float>& grid, int rows, int cols, std::vector<float>& out);

std::string ascii_digit(const std::vector<float>& pixels);
