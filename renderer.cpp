#include "renderer.hpp"
#include <algorithm>
#include <utility>

void draw_grid(sf::RenderWindow& window, int CELL_SIZE)
{
	int y = 0;

	while (y < window.getSize().y)
	{
		for (int x = 0; x <= window.getSize().x; x += CELL_SIZE)
		{
			sf::RectangleShape cell({ static_cast<float>(CELL_SIZE), static_cast<float>(CELL_SIZE) });
			cell.setPosition({ static_cast<float>(x), static_cast<float>(y) });
			cell.setFillColor(sf::Color::Black);
			cell.setOutlineThickness(1.f);
			cell.setOutlineColor(sf::Color(100, 100, 100));

			window.draw(cell);
		}

		y += CELL_SIZE;
	}
}

void paint(sf::RenderWindow& window, int CELL_SIZE, int col, int row, int& lastCol, int& lastRow, std::unordered_set<int>& grid, bool is_drawing)
{
	int cols = window.getSize().x / CELL_SIZE;
	int rows = window.getSize().y / CELL_SIZE;

	if (col >= 0 && row >= 0 && col < cols && row < rows)
	{
		// primer punto del trazo
		if (lastCol == -1 || lastRow == -1)
		{
			grid.insert(row * cols + col);
		}
		else
		{
			int x0 = lastCol;
			int y0 = lastRow;
			int x1 = col;
			int y1 = row;

			int dx = std::abs(x1 - x0);
			int dy = -std::abs(y1 - y0);

			int sx = x0 < x1 ? 1 : -1;
			int sy = y0 < y1 ? 1 : -1;

			int error = dx + dy;

			while (true)
			{
				if (x0 >= 0 && y0 >= 0 && x0 < cols && y0 < rows)
					grid.insert(y0 * cols + x0);

				if (x0 == x1 && y0 == y1)
					break;

				int e2 = 2 * error;

				if (e2 >= dy)
				{
					error += dy;
					x0 += sx;
				}

				if (e2 <= dx)
				{
					error += dx;
					y0 += sy;
				}
			}
		}

		if (is_drawing == 1)
		{
			lastCol = col;
			lastRow = row;
		}
	}

	for (int id : grid)
	{
		int currentRow = id / cols;
		int currentCol = id % cols;

		float x = static_cast<float>(currentCol * CELL_SIZE);
		float y = static_cast<float>(currentRow * CELL_SIZE);

		sf::RectangleShape cell({ static_cast<float>(CELL_SIZE),static_cast<float>(CELL_SIZE) });

		cell.setPosition({ x, y });
		cell.setFillColor(sf::Color::White);
		cell.setOutlineThickness(1.f);
		cell.setOutlineColor(sf::Color(100, 100, 100));

		window.draw(cell);
	}
}

sf::Image get_28x28_Image(const std::unordered_set<int>& painted, sf::RenderWindow& window, int CELL_SIZE)
{
	if (CELL_SIZE <= 0) return preprocess_digit(sf::Image{});
	int width = window.getSize().x / CELL_SIZE;
	int height = window.getSize().y / CELL_SIZE;
	if (width == 0 || height == 0) return preprocess_digit(sf::Image{});
	sf::Image image({ static_cast<unsigned int>(width), static_cast<unsigned int>(height) }, sf::Color::Black);

	for (int pixel : painted)
	{
		if (pixel >= 0 && pixel < width * height)
			image.setPixel({ static_cast<unsigned int>(pixel % width), static_cast<unsigned int>(pixel / width) }, sf::Color::White);
	}
	return preprocess_digit(image);
}

void render_loop(neural_network& nn)
{
	unsigned int WINDOW_SIZE = 800;
	int CELL_SIZE = 20.f;
	int col = -1;
	int row = -1;
	int lastCol = -1;
	int lastRow = -1;
	bool is_drawing = 0;
	bool clear = 0;
	bool end = 0;
	std::unordered_set<int> painted;

	sf::RenderWindow window(sf::VideoMode({ WINDOW_SIZE, WINDOW_SIZE }), "Blackboard", sf::Style::Close);

	std::cout << "- If you press CTR+Z, it cleans all the blackboard.\n- If you press CTR+Enter, the drawing will send it to the Neural Network.";

	while (window.isOpen() && !end)
	{
		if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
		{
			sf::Vector2i mousePos = sf::Mouse::getPosition(window);

			col = mousePos.x / CELL_SIZE;
			row = mousePos.y / CELL_SIZE;

			is_drawing = 1;
		}

		while (const std::optional event = window.pollEvent())
		{
			if (event->is<sf::Event::Closed>()) window.close();

			if (const auto* mouseReleased = event->getIf<sf::Event::MouseButtonReleased>())
			{
				if (mouseReleased->button == sf::Mouse::Button::Left)
				{
					lastCol = -1;
					lastRow = -1;

					is_drawing = 0;
				}
			}

			if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>())
			{
				if (keyPressed->code == sf::Keyboard::Key::Z &&
					sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl))
				{
					clear = 1;
				}

				if (keyPressed->code == sf::Keyboard::Key::Enter &&
					sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl))
				{
					std::cout << "\n- Preparing the drawing...\n";

					sf::Image img28 = get_28x28_Image(painted, window, CELL_SIZE);
					auto input = image_to_vector(img28);
					if (std::none_of(input.begin(), input.end(), [](float value) { return value > 0.0f; }))
					{
						std::cout << "- Draw a digit before submitting.\n";
						continue;
					}

					int label;

					while (true)
					{
						std::cout << "- Which number have you drawn?\n" << std::flush;
						if (!(std::cin >> label) || label > 9 || label < 0) std::cout << "- Stop typing shit, just write the correct number, it's not that hard.\n\n";
						else break;

						std::cin.clear();
						std::cin.ignore(1000, '\n');
					}

					// Store the exact processed vector that State::drawing passes to prediction.
					nn.images.push_back({ std::move(input), label });

					end = 1;
				}
			}
		}

		window.clear();

		if (clear)
		{
			painted.clear();
			clear = 0;

			lastCol = -1;
			lastRow = -1;

			col = -1;
			row = -1;

			is_drawing = 0;
		}

		draw_grid(window, CELL_SIZE);
		paint(window, CELL_SIZE, col, row, lastCol, lastRow, painted, is_drawing);

		window.display();
	}
}
