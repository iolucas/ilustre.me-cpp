#include <opencv2\core\core.hpp>

using namespace cv;
using namespace std;

#pragma once
class PersonFace
{
	Rect face;

    vector<Rect> mouths;
    Rect evaluatedMouth;
	vector<Rect> eyes;
    
    vector<Rect> evaluatedEyes;

	bool isValid;

    float faceLineSlope;
    float faceLineOffset;

public:
	PersonFace(Rect faceRect);
	~PersonFace(void);

	bool IsValid(void);
	vector<float> GetFaceLineData(void);
	Rect GetFace(void);
	void AddMouth(Rect mouth);
	Rect GetMouth(void);
	void AddEye(Rect eye);
	vector<Rect> GetEyes(void);
	Rect GetNose(void);
	void Evaluate(void);
};

