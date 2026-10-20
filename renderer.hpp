#pragma once
#include <vector>
#include <SFML/Graphics.hpp>
#include <unordered_set>
#include <iostream>
#include "neural_network.hpp"
#include "digit_preprocessing.hpp"

void render_loop(neural_network& nn);

sf::Image get_28x28_Image(const std::unordered_set<int>& painted, sf::RenderWindow& window, int CELL_SIZE);

void paint(sf::RenderWindow& window, int CELL_SIZE, int col, int row, int& lastCol, int& lastRow, std::unordered_set<int>& grid, bool is_drawing);

void draw_grid(sf::RenderWindow& window, int CELL_SIZE);
