#pragma once
#include <vector>
#include <string>
#include "opencv2/imgproc.hpp"

class ImageHelpers
{
	public:
		static std::vector<float> loadImage(const std::string& filename, int sizeX = 224, int sizeY = 224);
		static std::vector<std::string> loadLabels(const std::string& filename);
};
