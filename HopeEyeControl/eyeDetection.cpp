#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String faceCascadeName = "haarcascade_frontalface_alt.xml";
String eyesCascadeName = "haarcascade_eye_alt.xml";
String eyepairCascadeName = "haarcascade_mcs_eyepair_big.xml";
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
CascadeClassifier eyepairCascade;
String testFilename = "MorganTest.avi";
int usbWebcamIndex = 1;
String captureWindowName = "Capture - Face detection";

/** @function main */
int main(void)
{
	VideoCapture capture = VideoCapture(testFilename);
	Mat frame;

	//-- 1. Load the cascades
	if (!faceCascade.load(faceCascadeName)) 
	{ 
		printf("--(!)Error loading face cascade\n"); 
		return -1; 
	};
	if (!eyesCascade.load(eyesCascadeName)) 
	{ 
		printf("--(!)Error loading eyes cascade\n"); 
		return -1; 
	};
	if (!eyepairCascade.load(eyepairCascadeName)) 
	{ 
		printf("--(!)Error loading eyes cascade\n"); 
		return -1; 
	};

	//-- 2. Read the video stream
	if (!capture.isOpened()) 
	{ 
		printf("--(!)Error opening video capture\n"); 
		return -1; 
	}

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Detect facial features and display result
		detectAndDisplay(frame);

		int c = waitKey(10);
		if ((char)c == 27) { break; } // escape
	}
	return 0;
}

/** @function detectAndDisplay 
Detects facial features in a given frame and displays the result
*/
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	//-- Convert frame to grayscale and increase contrast
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(150, 150));
	
	for (size_t i = 0; i < faces.size(); i++)
	{
		//-- Draw circle around face
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		//-- In each face, detect eye pair
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyepairs;
		int minEyepairSizeX = faces[0].width / 2;
		int minEyepairSizeY = faces[0].height / 6;
		int maxEyepairSizeX = faces[0].width;
		int maxEyepairSizeY = faces[0].height / 3;
		eyepairCascade.detectMultiScale(faceROI, eyepairs, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(minEyepairSizeX, minEyepairSizeY), Size(maxEyepairSizeX, maxEyepairSizeY));
		
		for (size_t j = 0; j < eyepairs.size(); j++)
		{
			//-- Draw box around eye pair position
			rectangle(frame, faces[i].tl() + eyepairs[j].tl(), faces[i].tl() + eyepairs[j].br(), Scalar(255, 255, 0), 4, 8, 0);
		}

		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		int minEyeSizeX = faces[0].width / 5;
		int minEyeSizeY = faces[0].height / 6;
		int maxEyeSizeX = faces[0].width / 3;
		int maxEyeSizeY = faces[0].height / 4;
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(minEyeSizeX, minEyeSizeY), Size(maxEyeSizeX, maxEyeSizeY));
		//-- Draw circle around eye positions
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

	}

	//-- Show what you got
	imshow(captureWindowName, frame);
}