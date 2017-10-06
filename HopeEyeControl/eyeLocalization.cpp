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

	}
	return 0;
}
// function to localize the iris
cv::Vec3f getEyeball(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
	std::vector<int> sums(circles.size(), 0);
	for (int y = 0; y < eye.rows; y++)
	{
		uchar *ptr = eye.ptr<uchar>(y);
		for (int x = 0; x < eye.cols; x++)
		{
			int value = static_cast<int>(*ptr);
			for (int i = 0; i < circles.size(); i++)
			{
				cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
				int radius = (int)std::round(circles[i][2]);
				if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
				{
					sums[i] += value;
				}
			}
			++ptr;
		}
	}
	int smallestSum = 9999999;
	int smallestSumIndex = -1;
	for (int i = 0; i < circles.size(); i++)
	{
		if (sums[i] < smallestSum)
		{
			smallestSum = sums[i];
			smallestSumIndex = i;
		}
	}
	return circles[smallestSumIndex];
}

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

// this function returns only the rectangle from the top right position as a focal point
Rect getRightmostEye(std::vector<cv::Rect> &eyes)
{
	int rightmost = 99999999;
	int rightmostIndex = -1;
	for (int i = 0; i < eyes.size(); i++)
	{
		if (eyes[i].br().x < rightmost)
		{
			rightmost = eyes[i].br().x;
			rightmostIndex = i;
		}
	}
	return eyes[rightmostIndex];
}

std::vector<cv::Point> centers;
cv::Point stabilize(std::vector<cv::Point> &points, int windowSize)
{
	float sumX = 0;
	float sumY = 0;
	int count = 0;
	for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
	{
		sumX += points[i].x;
		sumY += points[i].y;
		++count;
	}
	if (count > 0)
	{
		sumX /= count;
		sumY /= count;
	}
	return cv::Point(sumX, sumY);
}

// function to detect the face and eyes
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
	faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50, 150));
	if (faces.size() == 0) return; // none face was detected
	//cv::Mat faceROI = frame(faces[0]); // crop the face

	for (size_t i = 0; i < faces.size(); i++)
	{
		//-- draw rectangle around face
		rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);

		//-- initializing an array to hold the face image afater thresholding
		cv::Mat faceROI = frame_gray(faces[i]);
		
		// detect eyes from face
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			//-- Draw box around eyes position
			rectangle(frame, faces[i].tl() + eyes[j].tl(), faces[i].tl() + eyes[j].br(), cv::Scalar(0, 255, 255), 2);
		}
		cv::Rect eyeRect = getLeftmostEye(eyes);
		cv::Mat eye = faceROI(eyeRect); // crop the leftmost eye
		cv::equalizeHist(eye, eye);
		std::vector<cv::Vec3f> circles;
		cv::HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
		if (circles.size() > 0)
		{
			cv::Vec3f eyeball = getEyeball(eye, circles);
			cv::Point center(eyeball[0], eyeball[1]);
			centers.push_back(center);
			center = stabilize(centers, 5); // we are using the last 5
			int radius = (int)std::round(circles[i][2]);
			cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
			cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
		}
		cv::imshow("Eye", eye);
	
	}


}

**/
	



	

