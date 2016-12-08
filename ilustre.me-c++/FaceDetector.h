#include <opencv2\objdetect\objdetect.hpp>

#include "PersonFace.h"

using namespace cv;

#pragma once
class FaceDetector
{
	vector<CascadeClassifier> frontFaceCascades;
    vector<CascadeClassifier> eyesCascades;
    vector<CascadeClassifier> nosesCascades;
	vector<CascadeClassifier> mouthsCascades;

public:
	FaceDetector(void);
	~FaceDetector(void);

	vector<PersonFace> Detect(Mat);
};

