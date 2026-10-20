



/*
#include <SFML/Graphics.hpp>
#include <unordered_set>
#include <iostream>

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

		sf::RectangleShape cell({static_cast<float>(CELL_SIZE),static_cast<float>(CELL_SIZE)});

		cell.setPosition({ x, y });
		cell.setFillColor(sf::Color::White);
		cell.setOutlineThickness(1.f);
		cell.setOutlineColor(sf::Color(100, 100, 100));

		window.draw(cell);
	}
}

sf::Image get_28x28_Image(const std::unordered_set<int>& painted, sf::RenderWindow& window, int CELL_SIZE)
{
	int width = window.getSize().x / CELL_SIZE;
	int height = window.getSize().y / CELL_SIZE;

	std::vector<std::uint8_t> pixels(width * height, 0);

	for (int pixel : painted)
	{
		if (pixel >= 0 && pixel < pixels.size())
			pixels[pixel] = 255;
	}

	sf::Image image({static_cast<unsigned int>(width),static_cast<unsigned int>(height)}, sf::Color::Black);

	for (int i = 0; i < pixels.size(); ++i)
	{
		int row = i / width;
		int col = i % width;

		std::uint8_t value = pixels[i];

		image.setPixel({static_cast<unsigned int>(col), static_cast<unsigned int>(row)}, sf::Color(value, value, value));
	}

	sf::Texture texture(image);
	texture.setSmooth(true);

	sf::Sprite sprite(texture);

	float scaleX = 28.f / static_cast<float>(width);
	float scaleY = 28.f / static_cast<float>(height);

	sprite.setScale({ scaleX, scaleY });
	sprite.setPosition({ 0.f, 0.f });

	sf::RenderTexture renderTexture({ 28, 28 });
	renderTexture.clear(sf::Color::Black);
	renderTexture.draw(sprite);
	renderTexture.display();

	return renderTexture.getTexture().copyToImage();
}

std::vector<float> image_to_vector(const sf::Image& image)
{
	std::vector<float> input;
	input.reserve(28 * 28);

	for (unsigned int y = 0; y < 28; ++y)
	{
		for (unsigned int x = 0; x < 28; ++x)
		{
			sf::Color color = image.getPixel({ x, y });

			float normalized = color.r / 255.0f;

			input.push_back(normalized);
		}
	}

	return input;
}

void render_loop()
{
	unsigned int WINDOW_SIZE = 800;
	int CELL_SIZE = 20.f;
	int col = -1;
	int row = -1;
	int lastCol = -1;
	int lastRow = -1;
	bool is_drawing = 0;
	bool clear = 0;
	std::unordered_set<int> painted;

	sf::RenderWindow window(sf::VideoMode({ WINDOW_SIZE, WINDOW_SIZE }), "Blackboard", sf::Style::Close);

	std::cout << "- If you press CTR+Z, it cleans all the blackboard.\n- If you press CTR+Enter, the drawing will send it to the Neural Network.";

	while (window.isOpen())
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
					image_to_vector(img28);


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
*/