#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

using ContourPair = std::pair<std::shared_ptr<Mat>, std::vector<Rect>>;

class Enhancement
{
public:
	// IO
	static void InitFromFile(string image_path);
	static void InitFromMat(Mat& image_data);
	static std::shared_ptr<Enhancement> GetInstancePtr();
	static std::shared_ptr<Mat> GetGrayImage(Mat& img, bool show);
	static string BuildFullPathname(string directory, string filename, string ext);
	static vector<string> ImagesPathFromDirectory(string directory, vector<string> types);

	std::shared_ptr<Mat> GetImage();

	// PRE-PROCESSING
	Mat EqualizeHistogram(bool show);
	Mat EqualizeClahe(bool show);
  	std::shared_ptr<Mat> GaussianFilter(Mat& img, bool show);
	std::shared_ptr<Mat> GammaCorrection(Mat& img, float gamma, bool show);
	std::pair<std::shared_ptr<Mat>, bool> ComputeHistogram(Mat& img, bool show);
  
	std::shared_ptr<Mat> MatchPattern(std::shared_ptr<Mat> baseImg, Mat& pattern, int method);
	ContourPair ExtractContorsBox(bool show);
	

private:
	Enhancement(string image_path);
	Enhancement(Mat& image_data);

	static std::shared_ptr<Enhancement> InstancePtr;
	std::shared_ptr<Mat> image_ptr;
};




