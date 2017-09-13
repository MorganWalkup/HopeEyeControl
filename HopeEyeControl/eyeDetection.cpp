#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>


/** Function Headers */
void detectAndDisplay(cv::Mat frame);
bool detectIris(cv::Mat eyeROI);

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
			detectIris(faceROI(eyes[j]));
			//imshow("eyeball",faceROI(eyes[j]));
		}

	}

	//-- Show what you got
	imshow(captureWindowName, frame);
}

bool detectIris(cv::Mat eyeROI)
{
	// Read image
	cv::Mat eye = eyeROI;

	// Thresholding
	//cv::threshold(eye, eye, 25, 255, cv::THRESH_BINARY);

	// Contrast and brightness adjustment
	eye.convertTo(eye,-1,3.5,-0.3);

	// Remove isolated pixels
	cv::medianBlur(eye, eye, 7);

	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = eye.cols * eye.rows / 30;
	params.maxArea = eye.cols * eye.rows / 4;

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

	if (numKeypoints > 1)
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
	imshow("keypoints", eye_with_keypoints);

	return true;
}