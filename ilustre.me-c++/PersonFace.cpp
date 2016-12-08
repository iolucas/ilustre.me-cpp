#include "PersonFace.h"
#include "PointGenerator.h"

#include <opencv2\core\core.hpp>

#include <iostream>

using namespace cv;
using namespace std;

PersonFace::PersonFace(Rect faceRect)
{
	face = faceRect;

    mouths = vector<Rect>();
    evaluatedMouth = Rect(0,0,0,0);

    eyes = vector<Rect>();
    evaluatedEyes = vector<Rect>();

	isValid = false;

    faceLineSlope = 0;
    faceLineOffset = 0;
}

bool PersonFace::IsValid(void) {
	return isValid;
}

vector<float> PersonFace::GetFaceLineData(void) {
	vector<float> retVec = vector<float>();

	retVec.push_back(faceLineSlope);
	retVec.push_back(faceLineOffset);

	return retVec;
}

Rect PersonFace::GetFace(void) { 
	return face;
}

void PersonFace::AddMouth(Rect mouth) {
	mouths.push_back(mouth);
}

Rect PersonFace::GetMouth(void) {
    return evaluatedMouth;
}

void PersonFace::AddEye(Rect eye) {
	eyes.push_back(eye);
}

vector<Rect> PersonFace::GetEyes(void) {
	vector<Rect> evEyes = vector<Rect>(evaluatedEyes);
    			
    return evEyes;
}

Rect PersonFace::GetNose(void) {
	PointGenerator faceLinePoint = PointGenerator::FromFactors(faceLineSlope, faceLineOffset);

    Size noseSize = Size(face.width / 7, face.height / 7);

    Point projNosePos = faceLinePoint.GetFromY(evaluatedMouth.y - noseSize.height);

    Rect createdNose = Rect(projNosePos.x - noseSize.width / 2, 
        	projNosePos.y - noseSize.height / 2, noseSize.width, noseSize.height);

    return createdNose;
}

void PersonFace::Evaluate(void) {

    	//Evaluate mouth
        evaluatedMouth = Rect(0, 0, 0, 0);

        //TODO must work a few on the mouth to choose the best one and proceed to histogram check for try to determinate skin color, eye color, hair color etc..

		for(int i = 0; i < mouths.size(); i++) {
			Rect mouth = mouths[i];

			//Check if the mouth is in the lower half of the face
			if(mouth.y < face.y + face.height / 2)
                continue;

			//Check if the current mouth is larger than the previous
            if(evaluatedMouth.width > mouth.width)
                continue;

            evaluatedMouth = mouth;
		}
        
        //Evaluate eyes
        evaluatedEyes = vector<Rect>();
        
        vector<Rect> rightCandidates = vector<Rect>();
        vector<Rect> leftCandidates = vector<Rect>();

		for(int i = 0; i < eyes.size(); i++) {
			Rect eye = eyes[i];

			//Ensure the eyes are in the upper half of the img region
            if(eye.y + eye.height / 2 > face.y + face.height / 2)
                continue;

            if(eye.x + eye.width / 2 < face.x + face.width / 2)
                rightCandidates.push_back(eye);
            else
                leftCandidates.push_back(eye);
		}


        //get centers for each side weighted by their areas
        float totalAreas = 0.0;
        float totalX = 0.0;
        float totalY = 0.0;

        if(rightCandidates.size() > 0) {
        
			for(int i = 0; i < rightCandidates.size(); i++) {
				Rect eye = rightCandidates[i];
				                
				float eyeArea = eye.width * eye.height;
                totalAreas += eyeArea;

                totalX += (eye.x + eye.width / 2) * eyeArea;
                totalY += (eye.y + eye.height / 2) * eyeArea;
			}            

            Point rightPoint = Point(totalX / totalAreas, totalY / totalAreas);

            float rightEyeSide = sqrt(totalAreas / rightCandidates.size());

            Rect rightEye = Rect(rightPoint.x - rightEyeSide / 2,
            	rightPoint.y - rightEyeSide/ 2, rightEyeSide, rightEyeSide);

            evaluatedEyes.push_back(rightEye);        
		}


        if(leftCandidates.size() > 0) {
        
            totalAreas = 0.0;
            totalX = 0.0;
            totalY = 0.0;

			for(int i = 0; i < leftCandidates.size(); i++) {
				Rect eye = leftCandidates[i];
            
                float eyeArea = eye.width * eye.height;
                totalAreas += eyeArea;

                totalX += (eye.x + eye.width / 2) * eyeArea;
                totalY += (eye.y + eye.height / 2) * eyeArea;
			}
            

            Point leftPoint = Point(totalX / totalAreas, totalY / totalAreas);

            float leftEyeSide = sqrt(totalAreas / leftCandidates.size());

            Rect leftEye = Rect(leftPoint.x - leftEyeSide / 2,
            	leftPoint.y - leftEyeSide / 2, leftEyeSide, leftEyeSide);

            evaluatedEyes.push_back(leftEye);        
		}


        //Reset flag and validate everything
        isValid = false;

        if(evaluatedEyes.size() == 2) {
        
            isValid = true;

            //Get the face line data

            Point eye1Center = Point(evaluatedEyes[0].x + evaluatedEyes[0].width / 2,
                evaluatedEyes[0].y + evaluatedEyes[0].height / 2);

            Point eye2Center = Point(evaluatedEyes[1].x + evaluatedEyes[1].width / 2,
                evaluatedEyes[1].y + evaluatedEyes[1].height / 2);

            float xOffset = (eye2Center.x - eye1Center.x) / 2;
            float yOffset = (eye2Center.y - eye1Center.y) / 2;

            Point eyeLineCenter = Point(eye1Center.x + xOffset, eye1Center.y + yOffset);

            //zeroDivFac = 1 if eye1Center.x == eye2Center.x else 0
			int zeroDivFac = eye1Center.x == eye2Center.x ? 1 : 0;

            //Generate face line slope and offset
            float aFact = (float)(eye1Center.y - eye2Center.y) / (float)(eye1Center.x - eye2Center.x + zeroDivFac);
			
			aFact = atan(aFact) + CV_PI / 2;
            aFact = tan(aFact);

            float bFact = eyeLineCenter.y - aFact * eyeLineCenter.x;

            faceLineSlope = aFact;
            faceLineOffset = bFact;

            //If the mouth is invalid, project a new based on the face line
            if(evaluatedMouth.width == 0) {
            
                PointGenerator faceLinePoint = PointGenerator::FromFactors(aFact, bFact);

                Point projMouthPos = faceLinePoint.GetFromY(face.y + face.height * 0.8);

                evaluatedMouth = Rect(projMouthPos.x - (face.width / 3) / 2, 
                	projMouthPos.y - (face.height / 5) / 2, face.width / 3, face.height / 5);
			}

		}

        if (evaluatedEyes.size() == 1 && evaluatedMouth.width > 0) {
        
            isValid = true;

            //Project the other eye based on the mouth

            //Get the bottom mouth coords
            Point mouthBottomCenter = Point(evaluatedMouth.width / 2 + evaluatedMouth.x,
                evaluatedMouth.y + evaluatedMouth.height);

            //get the facetop coords
            Point faceTopCenter = Point(face.width / 2 + face.x, face.y);

            //Apply an experimental correct factor to the values
            float correctFact = mouthBottomCenter.x - faceTopCenter.x;
            //correctFact = correctFact * 0.5

            mouthBottomCenter.x += correctFact;
            faceTopCenter.x -= correctFact;

            //Get the slope of the faceline

            //In case they are the same value, add a pixel to prevent division by 0
            //zeroDivFac = 1 if mouthBottomCenter.x == faceTopCenter.x else 0
			int zeroDivFac = mouthBottomCenter.x == faceTopCenter.x ? 1 : 0;

            float a = (mouthBottomCenter.y - faceTopCenter.y) / (mouthBottomCenter.x - faceTopCenter.x + zeroDivFac);

            //Get the offset of the face line
            float b = mouthBottomCenter.y - a * mouthBottomCenter.x;

            faceLineSlope = a;
            faceLineOffset = b;

            //Get the line function of the face
            PointGenerator faceLinePoint = PointGenerator::FromFactors(a, b);

            //Get the reference of the existing eye and its center point
            Rect eyeRef = evaluatedEyes[0];
            Point eyeCenter = Point(eyeRef.x + eyeRef.width / 2, eyeRef.y + eyeRef.height / 2);

            //Get the slope of the eye line (it must be normal to the face line, so we turn it Pi/2
			float aEyeFact = atan(a) + CV_PI / 2;
            aEyeFact = tan(aEyeFact);

            //Get the eye line offset
            float bEyeFact = eyeCenter.y - aEyeFact * eyeCenter.x;

            //Get the line function of the eye
            PointGenerator eyeLinePoint = PointGenerator::FromFactors(aEyeFact, bEyeFact);

            //Get the horizontal difference between the center of the existing eye and the face line
            float diff = faceLinePoint.GetFromY(eyeCenter.y).x - eyeCenter.x;

            //Get the project eye coords
            Point projEyePoint = eyeLinePoint.GetFromX(eyeCenter.x + diff * 2);
            
            //Get the project eye rectangle
            Rect projEyeRect = Rect(projEyePoint.x - eyeRef.width / 2, 
            	projEyePoint.y - eyeRef.height / 2, eyeRef.width, eyeRef.height);

            evaluatedEyes.push_back(projEyeRect);           
		}

        //If the face keep invalid, put the face line on the middle of the face square
        if (!isValid) {
            faceLineSlope = -face.height / 0.01;
            faceLineOffset = face.y - faceLineSlope * face.x + face.width / 2;
		}
    	
}

PersonFace::~PersonFace(void)
{
}
