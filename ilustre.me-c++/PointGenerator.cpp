#include "PointGenerator.h"

#include <opencv2\core\core.hpp>

using namespace cv;

PointGenerator::PointGenerator(float a, float b)
{
	aFact = a;
	bFact = b;
}

PointGenerator PointGenerator::FromFactors(float a, float b) {
    //"Creates a new instance of PointGenerator using line a and b factors."
    return PointGenerator(a, b);
}

PointGenerator PointGenerator::FromPoints(Point p1, Point p2) {
	//"Creates a new instance of PointGenerator using two points p1 and p2."

    float x1 = p1.x;
    float y1 = p1.y;
    float x2 = p2.x;
    float y2 = p2.y;

	if(x1 == x2)
		x1 += 0.001;

    float a = (y1 - y2) / (x1 - x2);
    float b = y1 - a * x1;

    return PointGenerator(a, b);
}

Point PointGenerator::GetFromX(float xValue) {
    return Point(xValue, aFact * xValue + bFact);
}

Point PointGenerator::GetFromY(float yValue) {
    return Point((yValue - bFact) / aFact, yValue);
}


PointGenerator::~PointGenerator(void)
{
}
