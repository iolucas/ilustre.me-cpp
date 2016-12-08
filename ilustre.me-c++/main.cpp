#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "FaceExtractor.h"

#include "FaceDetector.h"

#include "PersonFace.h"

using namespace std;

int main() {


	cout << "Ilustre.me\n\n";
	cout << "Insert image location:\n";

	//String to store image location
	string imageLocation;

	string basePath = "D:\\DiskD\\Pictures\\";

	imageLocation = basePath + "7505667_f260.jpg";
	imageLocation = basePath + "duffs.jpg";

	//cin >> imageLocation;

	try {
		//Load image into a Mat object
		cv::Mat image = cv::imread(imageLocation);


		FaceDetector faceDetector = FaceDetector();

		vector<PersonFace> personFaces = faceDetector.Detect(image);

		cout << personFaces.size();

		for(int i = 0; i < personFaces.size(); i++) {
			PersonFace personFace = personFaces[i];

			personFace.Evaluate();

			if(personFace.IsValid()) {
				rectangle(image, personFace.GetFace(), Scalar(255,0,0,255), 3);
				rectangle(image, personFace.GetMouth(), Scalar(255,0,0,255), 1);
				rectangle(image, personFace.GetNose(), Scalar(255,0,0,255), 1);
				vector<Rect> eyes = personFace.GetEyes();
				rectangle(image, eyes[0], Scalar(255,0,0,255), 1);
				rectangle(image, eyes[1], Scalar(255,0,0,255), 1);
			} else
				cout << "\nInvalid\n";
		}

		cv::imshow("Ilustre.me", image);


		//Wait until close image window
		cv::waitKey(0);

		return 0;

	} catch(exception e) {
		//Catch and display exception	

		cout << "\n\nERROR:\n\n" << e.what() << "\n\n";
		system("pause");

		return 1;
	}	
}