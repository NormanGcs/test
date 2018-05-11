/*
	图片旋转
*/
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <iostream>
#include <cmath>
using namespace cv;


void nearestInterpolation(Mat &src, Mat &dst, float dx, float dy, double theta);
void bilinearRotate(Mat &src, Mat &dst, float dx, float dy, double theta);
int main()
{
	cv::Mat src = imread("D:/xitong/picture/rain.jpg");
	namedWindow("orginal");
	imshow("orginal", src);
	int srcwidth = src.cols;
	int srcheigh = src.rows;
	/*旋转角度*/
	double theta = 30.0f*3.1415926 / 180.0f;
	/*
		转换坐标原点到图像中
				∧y
				|
			0	|	1
				|
		--------O--------->x
				|
			2	|	3
				|
	*/
	float srcX[4], srcY[4];
	srcX[0] = (float)(-((srcwidth - 1) / 2));
	srcX[1] = (float)((srcwidth - 1) / 2);
	srcX[2] = (float)(-(srcwidth - 1) / 2);
	srcX[3] = (float)((srcwidth - 1) / 2);
	srcY[0] = (float)((srcheigh - 1) / 2);
	srcY[1] = (float)((srcheigh - 1) / 2);
	srcY[2] = (float)(-(srcheigh - 1) / 2);
	srcY[3] = (float)(-(srcheigh - 1) / 2);

	/*
		旋转后的图像坐标，此时坐标原点依然是旋转中心
	*/
	float dstX[4], dstY[4];
	for (int i = 0; i < 4; i++)
	{
		dstX[i] = cos(theta)*srcX[i] + sin(theta)*srcY[i];
		dstY[i] = -sin(theta)*srcX[i] + cos(theta)*srcY[i];
	}
	
	/*
		==>旋转后图像长宽
	*/
	int dstwidth = (max(fabs(dstX[3] - dstX[0]), fabs(dstX[2] - dstX[1])) + 0.5);
	int dstheigh = (max(fabs(dstY[3] - dstY[0]), fabs(dstY[2] - dstY[1])) + 0.5);

	/*Mat dst = Mat(Size(src.rows * 2, src.cols * 2), src.type(), Scalar::all(0));
	nearestInterpolation(src, dst, 0.5);*/
	Mat dst;
	dst.create(dstheigh, dstwidth, src.type());
	
	//需要平移的量
	float dx = -0.5*dstwidth*cos(theta) - 0.5*dstheigh*sin(theta) + 0.5*srcwidth;
	float dy = 0.5*dstwidth*sin(theta) - 0.5*dstheigh*cos(theta) + 0.5*srcheigh;
	
	//nearestInterpolation(src, dst, dx, dy, theta);
	bilinearRotate(src, dst, dx, dy, theta);
	waitKey();
	return 0;
}

/*
	最邻近内插旋转
*/
void nearestInterpolation(Mat &src, Mat &dst, float dx, float dy, double theta)
{
	int x, y;
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			x = cvFloor(float(j)*cos(theta) + float(i)*sin(theta) + dx);
			y = cvFloor(float(-j)*sin(theta) + float(i)*cos(theta) + dy);
			if ((x < 0) || (x >= src.cols) || (y < 0) || (y >= src.rows))
			{
				if (src.channels() == 3)
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				if (src.channels() == 1)
					dst.at<uchar>(i, j) = 0;
			}
			else
			{
				if (src.channels() == 3)
					dst.at<Vec3b>(i, j) = src.at<Vec3b>(y, x);
				if (src.channels() == 1)
					dst.at<uchar>(i, j) = src.at<uchar>(y, x);
			}
			
		}
	}
	namedWindow("最邻近内插旋转");
	imshow("最邻近内插旋转", dst);
}

/*
	双线性内插法旋转
*/
void bilinearRotate(Mat &src, Mat &dst, float dx, float dy, double theta)
{
	float fu, fv;
	int x, y;
	Vec3b point[4];
	uchar upoint[4];
	for (int j = 0; j < dst.rows; j++
		)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			fu = float(j)*cos(theta) + float(i)*sin(theta) + dx;
			fv = float(-j)*sin(theta) + float(i)*cos(theta) + dy;
			x = cvFloor(fu);
			y = cvFloor(fv);
			fu -= x;
			fv -= y;
			
			if ((x < 0) || (x >= src.cols-1) || (y < 0) || (y >= src.rows-1))
			{
				if (src.channels() == 3)
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				if (src.channels() == 1)
					dst.at<uchar>(i, j) = 0;
			}
			else
			{
				if (src.channels() == 3)
				{
					point[0] = src.at<Vec3b>(y, x);
					point[1] = src.at<Vec3b>(y + 1, x);
					point[2] = src.at<Vec3b>(y, x + 1);
					point[3] = src.at<Vec3b>(y + 1, x + 1);
					dst.at<Vec3b>(i, j) = (1 - fu)*(1 - fv)*point[0] + (1 - fu)*(fv)*point[1] + (1 - fv)*(fu)*point[2] + fu*fv*point[3];
				}
				
				if (src.channels() == 1)
				{
					upoint[0] = src.at<uchar>(y, x);
					upoint[1] = src.at<uchar>(y + 1, x);
					upoint[2] = src.at<uchar>(y, x + 1);
					upoint[3] = src.at<uchar>(y + 1, x + 1);
					dst.at<uchar>(i, j) = (1 - fu)*(1 - fv)*upoint[0] + (1 - fu)*(fv)*upoint[1] + (1 - fv)*(fu)*upoint[2] + fu*fv*upoint[3];
				}
			}
		}
	}
	namedWindow("双线性内插法旋转");
	imshow("双线性内插法旋转", dst);
}
