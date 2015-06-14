#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <queue>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

using namespace cv;
using namespace std;

class gridPixel
{
public:
	gridPixel(){this->totalCost = 0.0;}
	gridPixel(const Point2d& position, const double& totalCost, const unsigned int& id)
	{	this->position = position;
		this->totalCost = totalCost;
		this->id = id;
	}
	gridPixel(const Point2d& position, const vector<unsigned int>& neighbors, const double& totalCost, const int& parent, const unsigned int& id, const unsigned int& depth)
	{	this->position = position;
		this->neighbors = neighbors;
		this->totalCost = totalCost;
		this->parent = parent;
		this->id = id;
		this->depth = depth;
	}
	gridPixel(const gridPixel& object)
	{
		this->position = object.position;
		this->neighbors = object.neighbors;
		this->totalCost = object.totalCost;
		this->parent = object.parent;
		this->id = object.id;
		this->depth = object.depth;
	}
	void operator=(const gridPixel& object)
	{
		this->position = object.position;
		this->neighbors = object.neighbors;
		this->totalCost = object.totalCost;
		this->parent = object.parent;
		this->id = object.id;
		this->depth = object.depth;
	}
	~gridPixel(){}
	bool operator<(const gridPixel& object) const
	{
		//return object.totalCost < this->totalCost; // ascend
		return object.totalCost > this->totalCost; // descend
	}
	Point2d position;
	vector<unsigned int> neighbors;
	double totalCost;
	int parent;
	unsigned int id;
	unsigned int depth;
};

void clickFunction(int event, int x, int y, int flags, void* userdata)
{
	static unsigned int flag = 0;
	int* data = static_cast<int*>(userdata);
	if(event == EVENT_LBUTTONDOWN && flag < 2)
	{
		if(!flag) { data[0] = x; data[1] = y; flag++;}
		else {data[2] = x; data[3] = y; flag++;}
	}
}

template<typename T>
inline double getNorm(Point_<T> startPoint, Point_<T> endPoint)
{
	double x1 = (double)(startPoint.x);
	double y1 = (double)(startPoint.y);
	double x2 = (double)(endPoint.x);
	double y2 = (double)(endPoint.y);

	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

struct gridInfo
{
	unsigned int startId;
	unsigned int endId;
	gridInfo(unsigned int startId, unsigned int endId)
	{
		this->startId = startId;
		this->endId = endId;
	}
	gridInfo(){}
};

void getGridPixelsAndStructure(const Point2d& startPoint,
								const Point2d& endPoint,
								double step,
								double angleCoverage,
								unsigned iteratorIncrement,
								Mat& image,
								vector<gridPixel>& gridPixels,
								vector<gridInfo>& gridStructure)
{
	unsigned id = 0;
	unsigned depth = 0;

	//push source node
	gridPixel sourceNode;
	sourceNode.position = startPoint;
	sourceNode.id = id;
	sourceNode.depth = depth;
	sourceNode.parent = id;
	sourceNode.totalCost = 0.0;
	gridPixels.push_back(sourceNode);
	gridStructure.push_back(gridInfo(0,0));
	id++;
	depth++;

	//find end-points on both sides (it may be out of image bounds but LineIterator handles it)
	Point2d direction = endPoint - startPoint;
	double radius = getNorm<double>(startPoint,endPoint);
	double radiusEndpoints = radius / cos((angleCoverage/2.0)*M_PI/180.0);
	double angle = atan2(direction.y, direction.x) * 180 / M_PI;
	double angleEndpoint1 = angle + angleCoverage/2.0;
	double angleEndpoint2 = angle - angleCoverage/2.0;

	Point2d endPoint1;
	endPoint1.x = startPoint.x + radiusEndpoints * cos(angleEndpoint1*M_PI/180.0);
	endPoint1.y = startPoint.y + radiusEndpoints * sin(angleEndpoint1*M_PI/180.0);

	Point2d endPoint2;
	endPoint2.x = startPoint.x + radiusEndpoints * cos(angleEndpoint2*M_PI/180.0);
	endPoint2.y = startPoint.y + radiusEndpoints * sin(angleEndpoint2*M_PI/180.0);

	//show end-points
	(image.at<Vec3b>(startPoint.y, startPoint.x))[0] = 0;
	(image.at<Vec3b>(startPoint.y, startPoint.x))[1] = 0;
	(image.at<Vec3b>(startPoint.y, startPoint.x))[2] = 255;

	(image.at<Vec3b>(endPoint1.y, endPoint1.x))[0] = 0;
	(image.at<Vec3b>(endPoint1.y, endPoint1.x))[1] = 0;
	(image.at<Vec3b>(endPoint1.y, endPoint1.x))[2] = 255;

	(image.at<Vec3b>(endPoint2.y, endPoint2.x))[0] = 0;
	(image.at<Vec3b>(endPoint2.y, endPoint2.x))[1] = 0;
	(image.at<Vec3b>(endPoint2.y, endPoint2.x))[2] = 255;

	(image.at<Vec3b>(endPoint.y, endPoint.x))[0] = 0;
	(image.at<Vec3b>(endPoint.y, endPoint.x))[1] = 0;
	(image.at<Vec3b>(endPoint.y, endPoint.x))[2] = 255;
	namedWindow("Debug",CV_WINDOW_FREERATIO);
	imshow("Debug", image);
	waitKey(0);

	//iterate till end-point level
	for(double i = step; i < (1.0 - 1e-4); i+=step)
	{
		//cout << i << endl;
		Point2d currentEndpoint1 = startPoint + i * (endPoint1 - startPoint);
		Point2d currentEndpoint2 = startPoint + i * (endPoint2 - startPoint);
		LineIterator iterator(image,currentEndpoint1,currentEndpoint2,8);
		for(int j = 0; j < iterator.count;)
		{
			//cout << (iterator.pos()) << endl;
/*			image.at<Vec3b>(iterator.pos())[0] = 0;
			image.at<Vec3b>(iterator.pos())[1] = 0;
			image.at<Vec3b>(iterator.pos())[2] = 255;
			namedWindow("Debug",CV_WINDOW_FREERATIO);
			imshow("Debug", image);
			waitKey(0);*/

			gridPixel currentGridPixel;
			currentGridPixel.id = id;
			currentGridPixel.depth = depth;
			currentGridPixel.position = iterator.pos();
			currentGridPixel.parent = -1;
			currentGridPixel.totalCost = 0.0;
			gridPixels.push_back(currentGridPixel);
			id++;

			//increment iterator
			for(unsigned increment = 0; increment < iteratorIncrement; increment++){j++; ++iterator;}
		}
		gridStructure.push_back(gridInfo(gridStructure.at(gridStructure.size()-1).endId + 1,id-1));
		depth++;
	}


	//push sink node
	gridPixel sinkNode;
	sinkNode.position = endPoint;
	sinkNode.id = id;
	sinkNode.depth = depth;
	sinkNode.parent = -1;
	sinkNode.totalCost = 0.0;
	gridPixels.push_back(sinkNode);
	gridStructure.push_back(gridInfo(id,id));
}

double getLocalCost(Point2d startPosition, Point2d endPosition, Mat& gPbImage)
{
	double gPbSum = 0.0;
	LineIterator iterator(gPbImage, startPosition, endPosition, 8);
	for(int i = 0; i < iterator.count; i++, ++iterator)
	{
		//cout << "(int)gPbImage.at<char>(iterator.pos()): " << (int)gPbImage.at<uchar>(iterator.pos()) << endl;
		gPbSum += gPbImage.at<uchar>(iterator.pos());
	}
	gPbSum /= iterator.count;
	return gPbSum;
}

void solveDynamicProgramming(vector<gridPixel>& gridPixels, vector<gridInfo>& gridStructure, Mat& gPbImage)
{
/*		cout << gridStructure.size() << endl;
		for(int ss = 0; ss < gridStructure.size(); ss++)
		{
			cout << "startId: " << gridStructure.at(ss).startId << " " << "endId: " << gridStructure.at(ss).endId <<  endl;
		}
		for(int ss = 0; ss < gridPixels.size(); ss++)
		{
			cout << gridPixels.at(ss).id << " " << gridPixels.at(ss).depth << endl;

		}
		cout << gridPixels.size() << endl;*/


	//define a priority queue
	priority_queue<gridPixel> solutionQueue;
	solutionQueue.push(gridPixels.at(0));

	//maximize localgPb
	while(!(solutionQueue.empty()))
	{
		//cout << "1" << endl;
		gridPixel currentQueueElement = solutionQueue.top();
		solutionQueue.pop();
		//cout << "2" << endl;
		if(currentQueueElement.id == gridStructure.at(gridStructure.size()-1).startId)
			break;
		//cout << "3" << endl;
		unsigned currentQueueElementDepth = currentQueueElement.depth;
		for(unsigned index = gridStructure.at(currentQueueElementDepth+1).startId ;
				index <= gridStructure.at(currentQueueElementDepth+1).endId; index++)
		{
			//cout << "4" << endl;
			double localCost = getLocalCost(currentQueueElement.position, gridPixels.at(index).position, gPbImage);
			//cout << "5" << endl;
			double totalCost = (currentQueueElement.totalCost * currentQueueElementDepth + localCost) / (currentQueueElementDepth + 1);
			//cout << "6" << endl;
			//cout << "totalCost: " << totalCost << endl;
			if(gridPixels.at(index).totalCost < totalCost)
			{
				//cout << "7" << endl;
				gridPixels.at(index).totalCost = totalCost;
				gridPixels.at(index).parent = currentQueueElement.id;
				solutionQueue.push(gridPixels.at(index));
				//cout << "8" << endl;
			}
		}
	}
}




int main( int argc, char** argv )
{
	//read image from argument
	Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//read gPb image from argument
	Mat gPbImage = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
	//cout << gPbImage.rows << " " << gPbImage.cols << endl;

	//create a window and get two mouse clicks
	namedWindow("Figure", CV_WINDOW_FREERATIO);
	int mouseClicks[4];
	setMouseCallback("Figure",clickFunction,mouseClicks);
	imshow("Figure", image);
	waitKey(0);

	//keep start and end points
	Point2d startPoint(mouseClicks[0], mouseClicks[1]);
	Point2d endPoint(mouseClicks[2], mouseClicks[3]);
	double radius = getNorm<double>(startPoint,endPoint);

	//define parameters
	double step = 0.005;
	double angleCoverage = 30.0;
	unsigned iteratorIncrement = 2;

	//get gridPixels and gridStructure
	vector<gridPixel> gridPixels;
	vector<gridInfo> gridStructure;
	getGridPixelsAndStructure(startPoint,
							endPoint,
							step,
							angleCoverage,
							iteratorIncrement,
							image,
							gridPixels,
							gridStructure);


	//solve dynamic programming
	cout << "Start dynamic programming.." << endl;
	solveDynamicProgramming(gridPixels, gridStructure, gPbImage);
	cout << "Finished dynamic programming.." << endl;

	//traverse solution
	vector<int> parentIdVector;
	int parentId = gridPixels.at(gridPixels.size()-1).id;
	while(1)
	{
		//cout << "parentId: " << parentId << endl;
		parentIdVector.push_back(parentId);
		//cout << gridPixels.at(parentId).totalCost << endl;
		if(parentId == gridPixels.at(parentId).parent )
			break;
		parentId = gridPixels.at(parentId).parent;
	}

	//show result
	Mat gPbImageColor = Mat::zeros(gPbImage.rows, gPbImage.cols,CV_8UC3);
	for(int r = 0; r < gPbImage.rows; r++)
	{
		for(int c = 0; c < gPbImage.cols; c++)
		{
			gPbImageColor.at<Vec3b>(r,c)[0] = gPbImage.at<char>(r,c);
			gPbImageColor.at<Vec3b>(r,c)[1] = gPbImage.at<char>(r,c);
			gPbImageColor.at<Vec3b>(r,c)[2] = gPbImage.at<char>(r,c);
		}
	}
	for(int i = 0; i < gridPixels.size(); i++)
	{
		gPbImageColor.at<Vec3b>(gridPixels.at(i).position)[0] = 0;
		gPbImageColor.at<Vec3b>(gridPixels.at(i).position)[1] = 0;
		gPbImageColor.at<Vec3b>(gridPixels.at(i).position)[2] = 255;
	}
	namedWindow("Debug", CV_WINDOW_FREERATIO);
	imshow("Debug", gPbImageColor);
	waitKey(0);
	for(int i = 0; i < parentIdVector.size()-1; i++)
	{
		LineIterator iterator(gPbImage, gridPixels.at(parentIdVector.at(i)).position,gridPixels.at(parentIdVector.at(i+1)).position,8);
		for(int j = 0; j < iterator.count; j++, ++iterator)
		{
			gPbImageColor.at<Vec3b>(iterator.pos())[0] = 255;
			gPbImageColor.at<Vec3b>(iterator.pos())[1] = 0;
			gPbImageColor.at<Vec3b>(iterator.pos())[2] = 0;
		}
	}
	namedWindow("Debug", CV_WINDOW_FREERATIO);
	imshow("Debug", gPbImageColor);
	waitKey(0);







/*	cout << gridStructure.size() << endl;
	for(int ss = 0; ss < gridStructure.size(); ss++)
	{
		cout << "startId: " << gridStructure.at(ss).startId << " " << "endId: " << gridStructure.at(ss).endId <<  endl;
	}
	for(int ss = 0; ss < gridPixels.size(); ss++)
	{
		cout << gridPixels.at(ss).id << " " << gridPixels.at(ss).depth << endl;

	}
	cout << gridPixels.size() << endl;*/
















/*	//mark clicked points (note the order BGR)
	(image.at<Vec3b>(mouseClicks[1], mouseClicks[0]))[0] = 0;
	(image.at<Vec3b>(mouseClicks[1], mouseClicks[0]))[1] = 0;
	(image.at<Vec3b>(mouseClicks[1], mouseClicks[0]))[2] = 255;
	(image.at<Vec3b>(mouseClicks[3], mouseClicks[2]))[0] = 0;
	(image.at<Vec3b>(mouseClicks[3], mouseClicks[2]))[1] = 0;
	(image.at<Vec3b>(mouseClicks[3], mouseClicks[2]))[2] = 255;
	namedWindow("Figure",CV_WINDOW_FREERATIO);
	imshow("Figure",image);
	waitKey(0);*/




	//cout << position[0] << " " << position[1] << " " << position[2] << " " << position[3] << endl;

	return 0;
}
