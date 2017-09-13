#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
bool detectIris(Mat eyeROI);

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
	vector<Rect> faces;
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
		vector<Rect> eyepairs;
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
		vector<Rect> eyes;
		int minEyeSizeX = faces[0].width / 5;
		int minEyeSizeY = faces[0].height / 6;
		int maxEyeSizeX = faces[0].width / 3;
		int maxEyeSizeY = faces[0].height / 4;
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(minEyeSizeX, minEyeSizeY), Size(maxEyeSizeX, maxEyeSizeY));
		//-- Draw box around eye positions
		for (size_t j = 0; j < eyes.size(); j++)
		{
			//-- Draw box around eye positions
			rectangle(frame, faces[i].tl() + eyes[j].tl(), faces[i].tl() + eyes[j].br(), Scalar(255, 0, 0), 4, 8, 0);
		
			//-- Locate iris within eye
			detectIris(faceROI(eyes[j]));
			//imshow("eyeball",faceROI(eyes[j]));
		}

	}

	//-- Show what you got
	imshow(captureWindowName, frame);
}

bool detectIris(Mat eyeROI)
{
	// Read image
	Mat eye = eyeROI;

	// Thresholding
	//threshold(eye, eye, 30, 225, cv::THRESH_BINARY);

	// Remove isolated pixels
	//medianBlur(eye, eye, 7);

	// Rename eye mat
	Mat im = eye;

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 100;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = im.cols * im.rows / 30;
	params.maxArea = im.cols * im.rows / 5;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.3f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.1f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01f;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs.
	vector<KeyPoint> keypoints;
	detector->detect(im, keypoints);

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;
	drawKeypoints(im, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);

	return true;
}