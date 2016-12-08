#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FaceExtractor.h"

#include "FaceDetector.h"

#include "PersonFace.h"
#include "PointGenerator.h"

using namespace std;

void drawIllustrations(Mat, vector<PersonFace>);

float RadianToDegree(float angle);
float DegreeToRadian(float angle);
float euclideanDistance(Point pt1, Point pt2);

Point getRectCenter(Rect rect);

int main() {


	cout << "Ilustre.me\n\n";
	cout << "Insert image location:\n";

	//String to store image location
	string imageLocation;

	string basePath = "D:\\DiskD\\Pictures\\";

	//imageLocation = basePath + "7505667_f260.jpg";
	//imageLocation = basePath + "duffs.jpg";

	cin >> imageLocation;

	try {
		//Load image into a Mat object
		cv::Mat image = cv::imread(imageLocation);


		FaceDetector faceDetector = FaceDetector();

		vector<PersonFace> personFaces = faceDetector.Detect(image);

		drawIllustrations(image, personFaces);
		
		
		/*cout << personFaces.size();

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
		}*/

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

float RadianToDegree(float angle) {
    return angle * (180.0 / CV_PI);
}

float DegreeToRadian(float angle) {
    return angle * (CV_PI / 180);
}

float euclideanDistance(Point pt1, Point pt2) {
    return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
}

Point getRectCenter(Rect rect) {
	return Point(rect.x + rect.width/2, rect.y + rect.height/2);
}


void drawIllustrations(Mat skyImage, vector<PersonFace> personFaces) {

	//COmpute img size
	Size imgSize = Size(skyImage.cols, skyImage.rows);

	for(int i = 0; i < personFaces.size(); i++) {
		PersonFace personFace = personFaces[i];

		personFace.Evaluate();

		if(!personFace.IsValid())
			continue;

		Rect faceRect = personFace.GetFace();
		Rect mouthRect = personFace.GetMouth();
		Rect noseRect = personFace.GetNose();
		vector<Rect> eyesRects = personFace.GetEyes();

        //Draw face division line
		vector<float> faceLineData = personFace.GetFaceLineData();
        PointGenerator faceLine = PointGenerator(faceLineData[0], faceLineData[1]);
		Point faceTopPoint = faceLine.GetFromY(faceRect.y);
        Point faceBottomPoint = faceLine.GetFromY(faceRect.y + faceRect.height);


		//line(skyImage, faceTopPoint,faceBottomPoint, Scalar(0,0,0),1);

		//Draw rect around the face
        //rectangle(skyImage, faceRect, Scalar(0,0,0));

        //Draw rect around the mouth
        //rectangle(skyImage, mouthRect, Scalar(0,0,0));

        //Draw rect around the nose
        //rectangle(skyImage, noseRect, Scalar(0,0,0));

        //Draw eyes rect and circles
        for(int j = 0; j < eyesRects.size(); j++) {
        	Rect eye = eyesRects[j];
            rectangle(skyImage, eye, Scalar(0,255,255));
        }

		
        //Get face feature angle
        float faceFeatureAngle = atan(faceLineData[0]);
        faceFeatureAngle = RadianToDegree(faceFeatureAngle);
        faceFeatureAngle += faceFeatureAngle > 0 ? -90 : 90;

        //Draw circle around face
        Point faceCenter = getRectCenter(faceRect);
		Size faceSize = Size(faceRect.width / 2, faceRect.height / 2);


		//ellipse(skyImage, faceCenter, faceSize, 0, 0, 360, Scalar(172, 203, 227));
		//ellipse(skyImage, faceCenter, faceSize, 0, 0, 360, Scalar(0,0,0));

    	//Draw face lateral boundaries lines
        //Detect right and left eye
        Rect rightEye, leftEye;
        if(eyesRects[0].x > eyesRects[1].x)
        {
            rightEye = eyesRects[1];
            leftEye = eyesRects[0];
        }
        else
        {
            rightEye = eyesRects[0];
            leftEye = eyesRects[1];
        }

        //get eye line generator
		PointGenerator eyeLines = PointGenerator::FromPoints(getRectCenter(rightEye), getRectCenter(leftEye));

        Point leftFacePoint = eyeLines.GetFromX(getRectCenter(leftEye).x + leftEye.width);
        Point rightFacePoint = eyeLines.GetFromX(getRectCenter(rightEye).x - rightEye.width);

        //circle(skyImage, leftFacePoint, 10, Scalar(0,255,0), -1);
        //circle(skyImage, rightFacePoint, 10, Scalar(255,0,0), -1);



        //Get line generators for each side of the face
        float faceLineSlope = faceLineData[0];

        //Left side
        float leftFaceSideOffset = leftFacePoint.y - leftFacePoint.x * faceLineSlope;
        PointGenerator leftFaceLine = PointGenerator(faceLineSlope, leftFaceSideOffset);

        Point startPointL = leftFaceLine.GetFromY(0);
        Point endPointL = leftFaceLine.GetFromY(imgSize.height);

        //Right side
        float rightFaceSideOffset = rightFacePoint.y - rightFacePoint.x * faceLineSlope;
        PointGenerator rightFaceLine = PointGenerator(faceLineSlope, rightFaceSideOffset);

        Point startPointR = rightFaceLine.GetFromY(0);
        Point endPointR = rightFaceLine.GetFromY(imgSize.height);

        //line(skyImage, startPointL, endPointL, Scalar(0,255,0));
        //line(skyImage, startPointR, endPointR, Scalar(255,0,0));

        //Draw mouth line
        //Put center on the top for the mouth stay in the middle of the mouth square
        Point mouthCenter = Point(mouthRect.x + mouthRect.width / 2, mouthRect.y);
        Size mouthSize = Size(mouthRect.width / 2, mouthRect.height / 2);


        Point mCenter = getRectCenter(mouthRect);

        //Get mouth line generator
        float aFactMouth = tan(atan(faceLineSlope) + CV_PI / 2);
        float bfactMouth = mCenter.y - mCenter.x * aFactMouth;
        PointGenerator mouthLine = PointGenerator(aFactMouth, bfactMouth);

        float leftFaceMouthCrossX = (bfactMouth - leftFaceSideOffset) / (faceLineSlope - aFactMouth);

        float rightFaceMouthCrossX = (bfactMouth - rightFaceSideOffset) / (faceLineSlope - aFactMouth);

        Point leftFaceMouthCross = mouthLine.GetFromX(leftFaceMouthCrossX);
        Point rightFaceMouthCross = mouthLine.GetFromX(rightFaceMouthCrossX);

        //Get face top line
        float afactTopFace = aFactMouth;   //use the mouth line since this uses the same slope
        float bfactTopFace = faceTopPoint.y - faceTopPoint.x * afactTopFace;
        PointGenerator faceTopLine = PointGenerator(afactTopFace, bfactTopFace);

        float leftTopFaceCrossX = (bfactTopFace - leftFaceSideOffset) / (faceLineSlope - afactTopFace);

        float rightTopFaceCrossX = (bfactTopFace - rightFaceSideOffset) / (faceLineSlope - afactTopFace);

        Point leftTopFaceCross = faceTopLine.GetFromX(leftTopFaceCrossX);
        Point rightTopFaceCross = faceTopLine.GetFromX(rightTopFaceCrossX);

        //circle(skyImage, leftTopFaceCross, 5, Scalar(0,128,0), -1);
        //circle(skyImage, rightTopFaceCross, 5, Scalar(0,128,0), -1);
        //circle(skyImage, leftFaceMouthCross, 5, Scalar(0,0,0), -1);
        //circle(skyImage, rightFaceMouthCross, 5, Scalar(0,0,0), -1);
        //circle(skyImage, faceBottomPoint, 5, Scalar(0,0,0), -1);

        Point facePointsMat[] = {
			leftTopFaceCross,
            rightTopFaceCross,
            rightFaceMouthCross,
            faceBottomPoint,
			leftFaceMouthCross
		};

        fillConvexPoly(skyImage, facePointsMat, 5, Scalar(172, 203, 227));

        float faceWidth = sqrt(pow(rightTopFaceCross.x - leftTopFaceCross.x, 2) + 
            pow(rightTopFaceCross.y - leftTopFaceCross.y, 2));

        Point hairCenter = Point(rightTopFaceCross.x + (leftTopFaceCross.x - rightTopFaceCross.x) / 2,
            rightTopFaceCross.y + (leftTopFaceCross.y - rightTopFaceCross.y) / 2);


        //Imgproc.circle(skyImage, hairCenter, 10, new Scalar(0,0,255), -1);

        ellipse(skyImage, hairCenter, Size(faceWidth/2, faceWidth/4),faceFeatureAngle, 0, -180, Scalar(0,0,0), -1);

        ellipse(skyImage, rightTopFaceCross, Size(faceWidth*0.75, faceWidth/5),faceFeatureAngle, -270, -360, Scalar(0,0,0), -1);

        ellipse(skyImage, leftTopFaceCross, Size(faceWidth*0.25, faceWidth/5),faceFeatureAngle, -180, -270, Scalar(0,0,0), -1);


        //Draw mouth line
        ellipse(skyImage, mouthCenter, mouthSize, faceFeatureAngle, -180, -360, Scalar(0,0,0), 2);

        Point p1 = faceTopLine.GetFromX(0);
        Point p2 = faceTopLine.GetFromX(imgSize.width);

        //Draw nose line
        Point noseCenter = Point(noseRect.x + noseRect.width / 2, noseRect.y + noseRect.height / 2);
        Size noseSize = Size(noseRect.width / 2, noseRect.height / 2);
        float noseAngle = atan(faceLineData[0]);
        noseAngle = RadianToDegree(noseAngle);
        ellipse(skyImage, noseCenter, noseSize, noseAngle, 0, 180, Scalar(0,0,0), 2);

        //Draw eyes ellipses
        for(int j = 0; j < eyesRects.size(); j++) {
        	Rect eye = eyesRects[j];

            Point eyeCenter = Point(eye.x + eye.width / 2, eye.y + eye.height / 2);
            Size elipseSize = Size(eye.width / 5, eye.height / 2);
            ellipse(skyImage, eyeCenter, elipseSize, faceFeatureAngle, 0, 360, Scalar(0,0,0), -1);
        }

        //Imgproc.line(skyImage, faceBottomPoint, new Point(imgSize.width / 2, imgSize.height), new Scalar(0,0,0));


        //Imgproc.circle(skyImage, faceTopPoint, 5, new Scalar(0,128,0), -1);
        //Imgproc.circle(skyImage, faceBottomPoint, 5, new Scalar(0,128,0), -1);


        float bodyHeight = euclideanDistance(faceTopPoint, faceBottomPoint);
        line(skyImage, faceBottomPoint, Point(faceBottomPoint.x, faceBottomPoint.y + bodyHeight),Scalar(0,0,0));

        float membersLength = bodyHeight / 2;

        Point bodyArmStart = Point(faceBottomPoint.x, faceBottomPoint.y + bodyHeight * 0.2);
        Point bodyLegStart = Point(faceBottomPoint.x, faceBottomPoint.y + bodyHeight);         
        
        line(skyImage, bodyArmStart, Point(bodyArmStart.x + membersLength * cos(CV_PI/4),
            bodyArmStart.y + membersLength * sin(CV_PI/4)),Scalar(0,0,0));

        line(skyImage, bodyArmStart, Point(bodyArmStart.x + membersLength * cos(CV_PI - CV_PI/4),
            bodyArmStart.y + membersLength * sin(CV_PI - CV_PI/4)), Scalar(0,0,0));

        line(skyImage, bodyLegStart,Point(bodyLegStart.x + membersLength * cos(CV_PI/4),
            bodyLegStart.y + membersLength * sin(CV_PI/4)), Scalar(0,0,0));

        line(skyImage, bodyLegStart,Point(bodyLegStart.x + membersLength * cos(CV_PI - CV_PI/4),
            bodyLegStart.y + membersLength * sin(CV_PI - CV_PI/4)), Scalar(0,0,0));
	}
}