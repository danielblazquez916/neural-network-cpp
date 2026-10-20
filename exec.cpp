#include <iostream>
#include "state.hpp"
#include <string>
#include <unordered_map>
#include "renderer.hpp"

int main()
{
	State s(32, 0.01f);

	std::cout << "testing coderabbit\n";

	while (true)
	{
		int state;

		while (true)
		{
			std::cout << "\n- Select state:\n* 0 -> Training Mode\n* 1 -> Testing Mode\n* 2 -> Drawing Mode\n* -1 -> Exit\n\n";

			if (!(std::cin >> state)) std::cout << "- Try again, don't be stupid pls\n\n";
			else if (state == -1) return 0;
			else if (state >= 0 && state <= 2) break;
			else std::cout << "- Try again, don't be stupid pls\n\n";

			std::cin.clear();
			std::cin.ignore(1000, '\n');
		}

		s.swap(static_cast<state_type>(state));
	}

	return 0;
}