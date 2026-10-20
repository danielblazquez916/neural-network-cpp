#pragma once

#include <SFML/Graphics/Image.hpp>
#include <vector>

sf::Image preprocess_digit(const sf::Image& image);

std::vector<float> image_to_vector(const sf::Image& image);
