#include "opencv2/opencv.hpp"
// openCV includes
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
// cpp includes
#include <iostream>
#include <stdio.h>

/** Function Headers */
//void detectAndDisplayFaceAndEyes(cv::Mat frame);
//void detectIris(cv::Mat& im_with_keypoints);

/** Global variables */
/**
cv::String faceCascadeName = "haarcascade_frontalface_alt.xml";
cv::String eyesCascadeName = "haarcascade_eye_alt.xml";
cv::String eyepairCascadeName = "haarcascade_mcs_eyepair_big.xml";

// initializing the cascade classifiers for the eyes and face
cv::CascadeClassifier faceCascade;
cv::CascadeClassifier eyesCascade;
cv::CascadeClassifier irisCascade;
cv::CascadeClassifier eyepairCascade;
cv::String captureWindowName = "Webcam Capture - Face and eye detection";

// creating vectors for the face, eye and pupil

std::vector<cv::Rect> faces;
std::vector<cv::Rect> eyepairs;
std::vector<cv::Rect> eyes;
std::vector<cv::Rect> pupil;


//int usbWebcamIndex = 1;
using namespace cv;

/** @function main */

/**
int main(void)
{
	cv::VideoCapture myCam(0);   // using videocapture class from opencv to load computer webcam
	cv::Mat frame;				 // initializing a video array for the webcam using the Mat class

								 //-- 1. Loading the cascades
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
		printf("--(!)Error loading eye pair cascade\n");
		return -1;
	};

	//-- 2. checking if webcam video is opened
	if (!myCam.isOpened())
	{
		printf("--(!)Error opening video capture\n");
		return -1;
	}
	while (myCam.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. calling the function to detect facial features and display result
		detectAndDisplayFaceAndEyes(frame);

		cv::imshow("captureWindowName", frame);
		int c = cv::waitKey(10);
		if ((char)c == 27) { break; } // escape
	    
		
		//if (cv::waitKey(30) >= 0) break;

	}
	return 0;
}

// getting left most eye
// this function returns only the rectangle from the top left position as a focal point
Rect getLeftmostEye(std::vector<cv::Rect> &eyes)
{
	int leftmost = 99999999;
	int leftmostIndex = -1;
	for (int i = 0; i < eyes.size(); i++)
	{
		if (eyes[i].tl().x < leftmost)
		{
			leftmost = eyes[i].tl().x;
			leftmostIndex = i;
		}
	}
	return eyes[leftmostIndex];
}

void detectAndDisplayFaceAndEyes(cv::Mat frame)
{
	Mat frame_gray;
	// Draw detected blobs/keypoints as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;

	//-- Convert frame to grayscale and increase contrast
	cv::cvtColor(frame, frame_gray, CV_BGR2GRAY); // convert image to grayscale
	equalizeHist(frame_gray, frame_gray);

	//-- Detect face from input image
	faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));
	if (faces.size() == 0) return; // none face was detected
	cv::Mat faceROI = frame(faces[0]); // crop the face

	for (size_t i = 0; i < faces.size(); i++)
	{
		//-- draw rectangle around face
		rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);

		//-- initializing an array to hold the face image afater thresholding
		cv::Mat faceROI = frame_gray(faces[i]);
		// initializing dimensions
		int minEyepairSizeX = faces[0].width / 2;
		int minEyepairSizeY = faces[0].height / 6;
		int maxEyepairSizeX = faces[0].width;
		int maxEyepairSizeY = faces[0].height / 3;

		// detecting the eye pair from the face region of interest
		eyepairCascade.detectMultiScale(faceROI, eyepairs, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(minEyepairSizeX, minEyepairSizeY), cv::Size(maxEyepairSizeX, maxEyepairSizeY));

		for (size_t j = 0; j < eyepairs.size(); j++)
		{
			//-- Draw box around eye pair position
			rectangle(frame, faces[i].tl() + eyepairs[j].tl(), faces[i].tl() + eyepairs[j].br(), cv::Scalar(255, 255, 0), 4, 8, 0);
		}

		// detect eyes from face
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			//-- Draw box around eyes position
			rectangle(frame, faces[i].tl() + eyes[j].tl(), faces[i].tl() + eyes[j].br(), cv::Scalar(0, 255, 255),2);
			//-- initializing an array of eye region of interest from the face region
			cv::Mat eyeROI = faceROI(eyes[j]);

			// Setup SimpleBlobDetector parameters.
			SimpleBlobDetector::Params params;

			// Change thresholds
			params.minThreshold = 10;
			params.maxThreshold = 200;

			// Filter by Area.
			params.filterByArea = true;
			params.minArea = 1500;

			// Filter by Circularity
			params.filterByCircularity = true;
			params.minCircularity = 0.1;

			// Filter by Convexity
			params.filterByConvexity = true;
			params.minConvexity = 0.87;

			// Filter by Inertia
			params.filterByInertia = true;
			params.minInertiaRatio = 0.01;

			// initializing vectors for keypoints
			std::vector<KeyPoint> keypoints;

			// intializing blob detector variable.
			cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
			// Detect blobs
			detector->detect(eyeROI, keypoints);
			// Show blobs
			drawKeypoints(eyeROI, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

			imshow("keypoints", im_with_keypoints);
		}
		
	}

}
**/