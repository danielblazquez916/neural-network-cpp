#include "utils.hpp"

std::pair<float, float> get_biggest(std::vector<float>& nums)
{
	float biggest = std::numeric_limits<float>::lowest();
	int index = 0;

	for (int i = 0; i < nums.size(); ++i)
	{
		if (nums[i] > biggest)
		{
			biggest = nums[i];
			index = i;
		}
	}
	return { biggest, index };
}

uint32_t read_uint32(std::ifstream& file)
{
	uint8_t bytes[4];
	file.read(reinterpret_cast<char*>(bytes), 4);

	return (bytes[0] << 24) |
		(bytes[1] << 16) |
		(bytes[2] << 8) |
		bytes[3];
}