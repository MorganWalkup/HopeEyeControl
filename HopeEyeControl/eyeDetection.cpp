#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <windows.h>


/** Function Headers */
void detectAndDisplay(cv::Mat frame);
bool detectIris(cv::Mat eyeROI, int lrFlag);
void moveMouse(int x, int y);

/** Global variables */
cv::String faceCascadeName = "haarcascade_frontalface_alt.xml";
cv::String eyesCascadeName = "haarcascade_eye_alt.xml";
cv::String eyepairCascadeName = "haarcascade_mcs_eyepair_big.xml";
cv::CascadeClassifier faceCascade;
cv::CascadeClassifier eyesCascade;
cv::CascadeClassifier eyepairCascade;
cv::String testFilename = "MorganTest.mp4";
int usbWebcamIndex = 1;
cv::String captureWindowName = "Capture - Face detection";
double minEyePositionX = 0.2;
double maxEyePositionX = 0.8;
double minEyePositionY = 0.4;
double maxEyePositionY = 0.6;

/** @function main */
int main(void)
{
	cv::VideoCapture capture = cv::VideoCapture(testFilename);
	cv::Mat frame;

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

		int c = cv::waitKey(10);
		if ((char)c == 27) { break; } // escape

		std::cin.ignore();

	}
	return 0;
}

/** @function detectAndDisplay 
Detects facial features in a given frame and displays the result
*/
void detectAndDisplay(cv::Mat frame)
{
	std::vector<cv::Rect> faces;
	cv::Mat frame_gray;

	//-- Convert frame to grayscale and increase contrast
	cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(150, 150));
	
	for (size_t i = 0; i < faces.size(); i++)
	{
		//-- Draw circle around face
		cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);

		//-- In each face, detect eye pair
		cv::Mat faceROI = frame_gray(faces[i]);
		std::vector<cv::Rect> eyepairs;
		int minEyepairSizeX = faces[0].width / 2;
		int minEyepairSizeY = faces[0].height / 6;
		int maxEyepairSizeX = faces[0].width;
		int maxEyepairSizeY = faces[0].height / 3;
		eyepairCascade.detectMultiScale(faceROI, eyepairs, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(minEyepairSizeX, minEyepairSizeY), cv::Size(maxEyepairSizeX, maxEyepairSizeY));
		
		for (size_t j = 0; j < eyepairs.size(); j++)
		{
			//-- Draw box around eye pair position
			rectangle(frame, faces[i].tl() + eyepairs[j].tl(), faces[i].tl() + eyepairs[j].br(), cv::Scalar(255, 255, 0), 4, 8, 0);
		}

		//-- In each face, detect eyes
		std::vector<cv::Rect> eyes;
		int minEyeSizeX = faces[0].width / 5;
		int minEyeSizeY = faces[0].height / 6;
		int maxEyeSizeX = faces[0].width / 3;
		int maxEyeSizeY = faces[0].height / 4;
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(minEyeSizeX, minEyeSizeY), cv::Size(maxEyeSizeX, maxEyeSizeY));
		//-- Draw box around eye positions
		for (size_t j = 0; j < eyes.size(); j++)
		{
			//-- Draw box around eye positions
			rectangle(frame, faces[i].tl() + eyes[j].tl(), faces[i].tl() + eyes[j].br(), cv::Scalar(255, 0, 0), 4, 8, 0);
		
			//-- Locate iris within eye
			if (eyes[j].tl().x < (faces[i].width / 2)) //If on left side of image (right eye)
				detectIris(faceROI(eyes[j]), 0); //Detect iris for right eye
			else
				detectIris(faceROI(eyes[j]), 1); //Detect iris for left eye
		}

	}

	//-- Show what you got
	imshow(captureWindowName, frame);
}

/** @@function detectIris
Detects the iris within an eye and displays it in a new window 
eyeROI: The matrix of the eye image
lrFlag: Flag for the eye type. 0 for right, 1 for left
*/
bool detectIris(cv::Mat eyeROI, int lrFlag)
{
	// Read image
	cv::Mat eye = eyeROI;

	// Thresholding
	//cv::threshold(eye, eye, 25, 255, cv::THRESH_BINARY);

	// Contrast and brightness adjustment
	double contrast = 2.5;
	double brightness = -0.3;
	double morganContrast = 3.5;
	double seanContrast = 2;
	double kehindeContrast = 3.5;
	eye.convertTo(eye,-1,contrast,brightness);

	// Remove isolated pixels
	cv::medianBlur(eye, eye, 7);

	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = eye.cols * eye.rows / 30.0f;
	params.maxArea = eye.cols * eye.rows / 4.0f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.25f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.05f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.3f;

	// Set up detector with params
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	// Detect blobs.
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(eye, keypoints);

	// Get the most central keypoint
	int numKeypoints = static_cast<int>(keypoints.size());

	if (numKeypoints == 0)
	{
		if (lrFlag == 0)
			imshow("Right Keypoints", eye);
		else if (lrFlag == 1)
			imshow("Left Keypoints", eye);

		return false;
	}
	else if (numKeypoints > 1)
	{
		int eyeMidpointX = eye.cols / 2;
		int eyeMidpointY = eye.rows / 2;
		int keyMidpointX = keypoints.at(0).pt.x;
		int keyMidpointY = keypoints.at(0).pt.y;

		float smallestDistance = sqrt(pow(eyeMidpointX - keyMidpointX, 2) + (eyeMidpointY - keyMidpointY, 2));
		cv::KeyPoint mostCentralKeypoint = keypoints.at(0);

		for (int i = 1; i < numKeypoints; i++) {
			keyMidpointX = keypoints.at(i).pt.x;
			keyMidpointY = keypoints.at(i).pt.y;
			float distance = sqrt(pow(eyeMidpointX-keyMidpointX,2) + (eyeMidpointY-keyMidpointY,2));

			if (distance < smallestDistance) {
				smallestDistance = distance;
				mostCentralKeypoint = keypoints.at(i);
			}
		}

		keypoints.clear();
		keypoints.push_back(mostCentralKeypoint);
	}

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	cv::Mat eye_with_keypoints;
	drawKeypoints(eye, keypoints, eye_with_keypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	if (lrFlag == 0) {
		imshow("Right Keypoints", eye_with_keypoints);
	}
	else if (lrFlag == 1) {
		imshow("Left Keypoints", eye_with_keypoints);

		// Move mouse to blob position for left eye
		int blobX = keypoints.at(0).pt.x;
		int blobY = keypoints.at(0).pt.y;
		int eyeWidth = eye.cols;
		int eyeHeight = eye.rows;

		double localEyePositionX = ((double)blobX) / eyeWidth;
		double localEyePositionY = ((double)blobY) / eyeHeight;

		/*if (localEyePositionX > maxEyePositionX)
		maxEyePositionX = localEyePositionX;
		if (localEyePositionX < minEyePositionX)
		minEyePositionX = localEyePositionX;
		if (localEyePositionY > maxEyePositionY)
		maxEyePositionY = localEyePositionY;
		if (localEyePositionY < minEyePositionY)
		minEyePositionY = localEyePositionY;*/

		double adjustedEyePositionX = localEyePositionX;
		double adjustedEyePositionY = localEyePositionY;

		int minEyeCoordX = minEyePositionX*eyeWidth;
		int maxEyeCoordX = maxEyePositionX*eyeWidth;
		int minEyeCoordY = minEyePositionY*eyeHeight;
		int maxEyeCoordY = maxEyePositionY*eyeHeight;

		if (maxEyeCoordX > minEyeCoordX)
			adjustedEyePositionX = ((double)blobX - minEyeCoordX) / (maxEyeCoordX - minEyeCoordX);

		if (maxEyeCoordY > minEyeCoordY)
			adjustedEyePositionY = ((double)blobY - minEyeCoordY) / (maxEyeCoordY - minEyeCoordY);

		double screenWidth = ::GetSystemMetrics(SM_CXSCREEN) - 1;
		double screenHeight = ::GetSystemMetrics(SM_CYSCREEN) - 1;
		printf("eyes (%f,%f)\n", screenWidth*adjustedEyePositionX, screenHeight*adjustedEyePositionY);
		moveMouse(screenWidth * adjustedEyePositionX, screenHeight * adjustedEyePositionY);
	}

	return true;
}

/* @@MouseMove
Moves the mouse to the given screen coordinates
*/
void moveMouse(int x, int y)
{
	double fScreenWidth = ::GetSystemMetrics(SM_CXSCREEN) - 1;
	double fScreenHeight = ::GetSystemMetrics(SM_CYSCREEN) - 1;
	double fx = x*(65535.0f / fScreenWidth);
	double fy = y*(65535.0f / fScreenHeight);
	INPUT  Input = { 0 };
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
	Input.mi.dx = fx;
	Input.mi.dy = fy;
	::SendInput(1, &Input, sizeof(INPUT));
}