#include <opencv2\core\core.hpp>

using namespace cv;

#pragma once
class PointGenerator
{
	float aFact, bFact;
public:
	PointGenerator(float, float);
	~PointGenerator(void);

	static PointGenerator FromFactors(float, float);
	static PointGenerator FromPoints(Point, Point);

	Point GetFromX(float);
	Point GetFromY(float);
};

