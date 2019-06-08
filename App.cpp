#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/ocl.hpp>
#include <Windows.h>



using namespace std;


cv::CascadeClassifier* faceCascade;
cv::CascadeClassifier* eyeCascade;
cv::Rect faceRoi;


int LoadClassifiers()
{
	try
	{
		eyeCascade = new cv::CascadeClassifier("res/haarcascade_eye.xml");
		faceCascade = new cv::CascadeClassifier("res/haarcascade_frontalface_alt.xml");
	}
	catch (const std::exception &e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
	return 0;
}

cv::Rect doubleRectSize(const cv::Rect &input_rect, const cv::Rect keep_inside)
{
	cv::Rect output_rect;
	// Double rect size
	output_rect.width = input_rect.width * 2;
	output_rect.height = input_rect.height * 2;

	// Center rect around original center
	output_rect.x = input_rect.x - input_rect.width / 2;
	output_rect.y = input_rect.y - input_rect.height / 2;

	// Handle edge cases
	if (output_rect.x < keep_inside.x) {
		output_rect.width += output_rect.x;
		output_rect.x = keep_inside.x;
	}
	if (output_rect.y < keep_inside.y) {
		output_rect.height += output_rect.y;
		output_rect.y = keep_inside.y;
	}

	if (output_rect.x + output_rect.width > keep_inside.width) {
		output_rect.width = keep_inside.width - output_rect.x;
	}
	if (output_rect.y + output_rect.height > keep_inside.height) {
		output_rect.height = keep_inside.height - output_rect.y;
	}

	return output_rect;
}

void DetectFeature(cv::Mat &frame, std::vector<cv::Rect> &faces, ::vector<cv::Rect> &eyes,   cv::CascadeClassifier* faceClassifier, cv::CascadeClassifier* eyeClassifier)
{ 
	cv::Mat grayScale;
	cv::cvtColor(frame, grayScale, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(grayScale, grayScale);
	faceClassifier->detectMultiScale(grayScale, faces, 1.1, 5, 0 , cv::Size(100,100));
	if (faces.size() != 0)
	{
		cv::Mat faceRegion = grayScale(faces[0]);
		cv::rectangle(frame, faces[0], cv::Scalar(0, 255, 255), 2);

		if (faces.size() > 0)
		{
			eyeClassifier->detectMultiScale(faceRegion, eyes, 1.1, 5, 0, cv::Size(20, 20));
			if (eyes.size() > 0)
			{
				for (int i = 0; i < eyes.size(); i++)
				{
					cv::rectangle(frame, faces[0].tl() + eyes[i].tl(), faces[0].tl() + eyes[i].br(), cv::Scalar(185, 0, 0), 2);

					cv::Mat eyeRegion = faceRegion(eyes[i]);
					std::vector<cv::Vec3f> circles;

					cv::HoughCircles(eyeRegion, circles, cv::HOUGH_GRADIENT, 1, eyeRegion.cols / 10, 250, 15, eyeRegion.rows / 10, eyeRegion.rows / 5);
					
					if (circles.size() > 0)
					{
						/*cv::Rect2d box = cv::selectROI(frame, false);*/
						//cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2, 1);
						//tracker->init(frame, box);
						for (int j = 0; j < circles.size(); j++)
						{
							//cv::drawMarker(frame, faces[0].tl() + eyes[i].tl() + cv::Point(circles[j][0], circles[j][1]), cv::Scalar(0, 0, 255), 2);
							cv::circle(frame, faces[0].tl() + eyes[i].tl() + cv::Point(circles[j][0], circles[j][1]), circles[j][2], cv::Scalar(0, 0, 255), 2);
						}
					}
					
				}
			}
		}
	}
	
}

void TrackFeature(cv::Mat &frame, cv::Mat &faceTemplate)
{

	cv::Mat result;

	cv::matchTemplate(frame, faceTemplate, result, cv::TM_CCOEFF);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	
	double minVal, maxVal;
	cv::Point  minLoc, maxLoc, topLeft, bottomRight;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	topLeft = maxLoc;

	bottomRight = cv::Point(topLeft.x + faceTemplate.cols, topLeft.y + faceTemplate.rows);

	cv::rectangle(frame, topLeft, bottomRight, CV_RGB(255, 0, 0), 0.5);

	faceTemplate = frame(cv::Rect(topLeft, bottomRight));

	/*cv::Mat result;
	int match_method = cv::TM_CCOEFF_NORMED;
	cv::matchTemplate(frame, faceTemplate, result, match_method);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	double minVal; double maxVal;
	cv::Point minLoc, maxLoc, matchLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)  
		matchLoc = minLoc;
	else 
		matchLoc = maxLoc;

	cv::rectangle
	(
		frame,
		matchLoc,
		cv::Point(matchLoc.x + faceTemplate.cols, matchLoc.y + faceTemplate.rows),
		CV_RGB(255, 0, 0),
		3
	);*/

}

int main()
{
	// main Frame Captured from the device.
	cv::Mat mainFrame;

	cv::VideoCapture capture(0);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	LoadClassifiers();
	
	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> eyes;
	cv::Mat faceTemplate;
	// Main Loop
	if (capture.isOpened())
	{
		while (cv::waitKey(1) != 'q')
		{
			capture >> mainFrame;

			cv::flip(mainFrame, mainFrame, 1);

			//if (faces.size() == 0)
			{
				DetectFeature(mainFrame, faces, eyes, faceCascade, eyeCascade);
				//cv::Rect rect = doubleRectSize(faces[0], cv::Rect(0,0, mainFrame.cols, mainFrame.rows));
				//faceTemplate = mainFrame(faces[0]);
			}
			//else
			{
				//TrackFeature(mainFrame, faceTemplate);
			}

			cv::imshow("EyePresent",mainFrame);
		}
	}
	

	delete(eyeCascade);
	delete(faceCascade);
	return 0;
}




