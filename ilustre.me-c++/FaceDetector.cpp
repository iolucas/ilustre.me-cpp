#include "FaceDetector.h"
#include "PersonFace.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>

using namespace cv;

#include <iostream>

FaceDetector::FaceDetector(void)
{
	//Init all the necessary cascades
    frontFaceCascades.push_back(CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml"));
		
    eyesCascades.push_back(CascadeClassifier("cascades/haarcascade_lefteye_2splits.xml"));
	eyesCascades.push_back(CascadeClassifier("cascades/haarcascade_righteye_2splits.xml"));

    nosesCascades.push_back(CascadeClassifier("cascades/Nose.xml"));
	nosesCascades.push_back(CascadeClassifier("cascades/haarcascade_mcs_nose.xml"));

	mouthsCascades.push_back(CascadeClassifier("cascades/Mouth.xml"));
}

vector<PersonFace> FaceDetector::Detect(Mat image) {
        vector<PersonFace> personFaces = vector<PersonFace>();

        Mat ugray = Mat();
			
		cvtColor(image, ugray, CV_BGR2GRAY);

        //normalizes brightness and increases contrast of the image
        //ugray = cv2.equalizeHist(ugray)
		equalizeHist(ugray, ugray);
                

        //for faceCascade in self.frontFaceCascades:
		for(int i = 0; i < frontFaceCascades.size(); i++) {
        
			CascadeClassifier faceCascade = frontFaceCascades[i];

			vector<Rect> detectedFaces = vector<Rect>();

			faceCascade.detectMultiScale(ugray,detectedFaces, 1.1, 3);

			for(int j = 0; j < detectedFaces.size(); j++) {
				Rect face = detectedFaces[j];

				//rectangle(image, personFace.GetMouth(), Scalar(255,0,0,255), 1);

                PersonFace personFace = PersonFace(face);

                //Get the region of interest on the faces
                //Mat faceRegion = new Mat(ugray, face);
                //roi = gray[y1:y2, x1:x2]
                
				//faceRegion = ugray[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
				Mat faceRegion = Mat(ugray, face);
                        
                //Detect eyes
				for(int k = 0; k < eyesCascades.size(); k++) {
					CascadeClassifier eyeCascade = eyesCascades[k];

					vector<Rect> detectedEyes = vector<Rect>();

                    eyeCascade.detectMultiScale(faceRegion, detectedEyes, 1.1, 3);

					//std::cout << "\nDetected Eyes: " << detectedEyes.size();

					//cv::imshow("ae", faceRegion);
					//cv::waitKey(0);

					//std::cout << "\nEyesDetected:\n" << detectedEyes.size();
                    
					for(int m = 0; m < detectedEyes.size(); m++) {
						Rect eye = detectedEyes[m];

						//Add the found eye with the offset of the entire img, since we detected it in the roi of the face
                        personFace.AddEye(Rect(eye.x + face.x, eye.y + face.y,eye.width, eye.height));
					}
				}
           
                //Detect mouths
				for(int k = 0; k < mouthsCascades.size(); k++) {
					CascadeClassifier mouthCascade = mouthsCascades[k];

                    vector<Rect> detectedMouths = vector<Rect>();
						
					mouthCascade.detectMultiScale(faceRegion, detectedMouths, 1.1, 3);

					for(int m = 0; m < detectedMouths.size(); m++) {
						Rect mouth = detectedMouths[m];

                        personFace.AddMouth(Rect(mouth.x + face.x, mouth.y + face.y, mouth.width, mouth.height));
					}
				}
                    
               
                //Detect noses (NOT USED)
                /*vector<Rect> noses = vector<Rect>();
                
				for(int k = 0; k < nosesCascades.size(); k++) {
					CascadeClassifier noseCascade = nosesCascades[k];
                
                    vector<Rect> detectedNoses = vector<Rect>();
						
					noseCascade.detectMultiScale(faceRegion, detectedNoses, 1.1, 10);

					for(int m = 0; m < detectedNoses.size(); m++) {
						Rect nose = detectedNoses[m];
                    

                        noses.push_back(Rect(nose.x + face.x, nose.y + face.y, nose.width, nose.height));
					}
				}*/

				//Push person reference to the vector
				personFaces.push_back(personFace);
			}
		}
                    
        return personFaces;
}


FaceDetector::~FaceDetector(void)
{
}
