#include <string>
#include <memory>
#include <experimental\filesystem>

#include "opencv2\opencv.hpp"
#include "Enhancement.h"
#include "TransformImage.h"

namespace fs = std::experimental::filesystem::v1;

using namespace std;
using namespace cv;
using cv::CLAHE;


std::shared_ptr<Enhancement> Enhancement::InstancePtr;

Enhancement::Enhancement(string image_ptr_path)
{
	auto mat = imread(image_ptr_path, -1); 
	image_ptr = std::make_shared<Mat>(mat);
}


Enhancement::Enhancement(Mat& image_ptr_data)
{
	image_ptr = std::shared_ptr<Mat>(new Mat(image_ptr_data));
}


std::shared_ptr<Enhancement> Enhancement::GetInstancePtr()
{
	if (InstancePtr == nullptr)
		throw std::runtime_error("Pointer not initialzed! Please call InitFromFile or InitFromMat...");
	return InstancePtr;
}


void Enhancement::InitFromFile(string image_ptr_path)
{
	if (InstancePtr == nullptr)
		InstancePtr = std::shared_ptr<Enhancement>(new Enhancement(image_ptr_path));
	
	InstancePtr->image_ptr = std::shared_ptr<Mat>(new Mat(imread(image_ptr_path, 1)));
}


void Enhancement::InitFromMat(Mat& image_ptr_data)
{
	if (InstancePtr == nullptr)
		InstancePtr = std::shared_ptr<Enhancement>(new Enhancement(image_ptr_data));
	else
		InstancePtr->image_ptr = std::shared_ptr<Mat>(new Mat(image_ptr_data));
}


vector<string> Enhancement::ImagesPathFromDirectory(string directory, vector<string> types)
{
	vector<string> filenames;

	string str_types;
	for (const auto & typ : types)
	{
		str_types += (typ + " ");
	}

	for (const auto& p : fs::directory_iterator(directory))
	{
		string filename = p.path().filename().string();
		size_t found = filename.find_last_of(".");
		size_t size = filename.size();

		if (found == string::npos)
			continue;

		string ext = filename.substr(found +1, size - 1);
		if (str_types.find(ext) != string::npos)
		{
			filename = Enhancement::BuildFullPathname(directory, filename, ext);
			filenames.emplace_back(filename);
		}
	}
	return std::move(filenames);
}


string Enhancement::BuildFullPathname(string directory, string filename, string ext)
{
	size_t found0;
	size_t found;
	
	if (directory.empty() || filename.empty() || ext.empty())
		return "";
	
	found0 = filename.find_last_of('/');
	if(found0 == string::npos)
		found0 = 0;

	auto basename = filename.substr(found0, filename.length()-1);
	found = basename.find_last_of('.');
	if (found != string::npos)
		basename = basename.substr(0, found);
	filename = string(pathJoin(directory.c_str(), basename.c_str()));
	if (ext.find('.') != string::npos)
		filename += ext;
	else
		filename = filename + "." + ext;
	return filename;
}


std::shared_ptr<Mat> Enhancement::GetGrayImage(Mat& img, bool show)
{
	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);
	return std::make_shared<Mat>(gray);
}

// PRE PROCESSING
Mat Enhancement::EqualizeHistogram(bool show = false)
{
	Mat imgEqualized;
	equalizeHist(*image_ptr.get(), imgEqualized);

	if (show)
	{
		Mat combined;
		hconcat(*image_ptr.get(), imgEqualized, combined);
		imshow("equalized histogram", combined);
		waitKey(20);
	}

	return imgEqualized;
}

Mat Enhancement::EqualizeClahe(bool show=false)
{
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(4);

	Mat imgEqualized;
	clahe->apply(*image_ptr.get(), imgEqualized);
	
	if (show)
	{
		Mat combined;
		hconcat(*image_ptr.get(), imgEqualized, combined);

		imshow("equalized clahe", combined);
		waitKey(20);
	}
	return imgEqualized;
}


std::shared_ptr<Mat> Enhancement::GetImage()
{
	return image_ptr;
}


std::shared_ptr<Mat> Enhancement::MatchPattern(std::shared_ptr<Mat> baseImg, Mat& pattern, int match_method)
{
	Mat match;
	Mat result;
	Mat mask;
	Mat img_display;
	std::shared_ptr<Mat> img_ptr = nullptr;
	pattern.copyTo(mask);

	baseImg.get()->copyTo(img_display);
	const auto& con_pair = ExtractContorsBox(false);
	img_ptr = con_pair.first;

	//img_ptr = image_ptr;
	if (img_ptr == nullptr || img_ptr.get() == nullptr)
		return img_ptr;
	int result_cols = img_ptr.get()->cols - pattern.cols + 1;
	int result_rows = img_ptr.get()->rows - pattern.rows + 1;
	if (result_cols <= 5 || result_rows <= 5)
		return img_ptr;

	result.create(result_rows, result_cols, CV_32FC1);
	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	if(method_accepts_mask)
	{
		matchTemplate(*img_ptr.get(), pattern, result, match_method, mask);
	}
	else
	{
		matchTemplate(*img_ptr.get(), pattern, result, match_method);
	}
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}
	auto match_point = Point(matchLoc.x + pattern.cols, matchLoc.y + pattern.rows);
	rectangle(img_display, matchLoc, match_point, Scalar::all(1), 2, 8, 0);
	rectangle(result, matchLoc, match_point, Scalar::all(0), 2, 8, 0);

	string window_name("Orig");
	namedWindow(window_name, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	resizeWindow(window_name, 600, 200);
	imshow(window_name, img_display);
	//imshow("REs", result);
	waitKey(2);

	return std::make_shared<Mat>(img_display);
}


ContourPair Enhancement::ExtractContorsBox(bool show = false)
{
	char *selname = nullptr;
	

	if (image_ptr == nullptr || image_ptr.get()->rows < 1)
		return std::make_pair(nullptr, std::vector<Rect>());
	
	Mat YCrCb;
	Mat HLS;
	Mat img_edges;

	Mat gray;
	Mat RGB;
	Mat thrYCrCb;
	Mat channels[3];
	std::shared_ptr<Mat> img_ptr;

	image_ptr.get()->copyTo(RGB);
	cvtColor(RGB, gray, CV_RGB2GRAY);
	const auto& hist_pair = ComputeHistogram(gray, false);
	if (hist_pair.second == false)
	{
		Mat tmp;
		img_ptr = GammaCorrection(gray, 0.8, true);
		//threshold(gray, gray, 90, 128, CV_THRESH_TRIANGLE);
	}
	else
		img_ptr = std::make_shared<Mat>(gray);
	
	cvtColor(*img_ptr.get(), RGB, CV_GRAY2RGB);
	img_ptr.get()->copyTo(gray);

	Mat neg_image = Mat(image_ptr.get()->rows, image_ptr.get()->cols, image_ptr.get()->type(), Scalar(0, 0, 0));
	Mat sub_mat = Mat(image_ptr.get()->rows, image_ptr.get()->cols, image_ptr.get()->type(), Scalar(1, 1, 1)) * 255;
	subtract(sub_mat, RGB, neg_image);

	Mat tmp, edges[3];
	Mat channels1[3];
	std::vector<Mat> contours;
	vector< Vec4i > hierarchy;
	std::vector<Rect> contoursRect;
	std::shared_ptr<Mat> gamma_ptr = nullptr;
	Mat neg_gray;
	cvtColor(neg_image, neg_gray, CV_RGB2GRAY);

	const auto& neg_pair = ComputeHistogram(neg_gray, false);
	if (hist_pair.second == false)
	{
		gamma_ptr = GammaCorrection(neg_gray, 1, false);
		//gamma_ptr.get()->copyTo(neg_gray);
	}
	else
		img_ptr = std::make_shared<Mat>(neg_gray);
	
	GaussianBlur(neg_gray, tmp, Size(3, 3), 0, 3);
	Mat ch[2] = { neg_gray, tmp };
	addWeighted(*gamma_ptr.get(), 1.5, tmp, -0.5, 90, tmp);
	merge(ch, 1, neg_gray);

	//channels1[0] = TransformImage::rotateImage(neg_gray, 90, 80, 90, 0, 0, 200, 200);
	//channels1[1] = TransformImage::rotateImage(neg_gray, 90, 120, 90, 0, 0, 200, 200);
	//channels1[2] = TransformImage::rotateImage(neg_gray, 90, 50, 90, 0, 0, 200, 200);

	cvtColor(RGB, HLS, CV_BGR2HLS);
	split(HLS, channels);
	GaussianBlur(channels[1], channels[1], Size(5, 5), 0, 2);
	if (hist_pair.second == true)
		threshold(neg_gray, neg_gray, 128, 255, CV_THRESH_BINARY_INV);

	Canny(neg_gray, neg_gray, 60, 255, 3, true);
	findContours(neg_gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	cvtColor(neg_gray, neg_image, CV_GRAY2RGB);
	Mat outImg = Mat(neg_gray.rows, neg_gray.cols, neg_gray.type(), Scalar(1)) * 255;

	int i = -1;
	for (const auto & contour : contours)
	{
		i += 1;
		auto rec = boundingRect(contour);
		
		if (hierarchy[i][2] < 2 || rec.width < 5 || rec.height < 5 || 
			rec.width > rec.height || rec.height >(rec.width / 2 + rec.height))
			continue;
		Mat patch;
		contoursRect.emplace_back(rec);
		rectangle(RGB, Point(rec.x, rec.y), Point(rec.x + rec.width, rec.y + rec.height), (100, 100, 100), 1);
		getRectSubPix(gray, rec.size(), Point(rec.x + (rec.width * 0.5), rec.y + (rec.height * 0.5)), patch);
		patch.copyTo(outImg(cv::Rect(rec.x, rec.y, patch.cols, patch.rows)));
	}

	cvtColor(outImg, outImg, CV_GRAY2RGB);

	if (show)
	{	
		Mat tmp;
		string window_name("Countours");
		namedWindow(window_name, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
		resizeWindow(window_name, 700, 280);
		
		Mat combined;
		hconcat(RGB, outImg, combined);
		imshow(window_name, combined);
	}

	auto const& out_ptr = std::make_shared<Mat>(outImg);
	return std::make_pair(out_ptr, contoursRect);
}


std::shared_ptr<Mat> Enhancement::GammaCorrection(Mat& img, float gamma, bool show)
{
	CV_Assert(img.data);

	// accept only char type matrices
	CV_Assert(img.depth() != sizeof(uchar));

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
	}

	Mat img_gamma = img.clone();
	const int channels = img_gamma.channels();
	switch (channels)
	{
		case 1:
		{
			MatIterator_<uchar> it, end;
			for (it = img_gamma.begin<uchar>(), end = img_gamma.end<uchar>(); it != end; it++)
				*it = lut[(*it)];

			break;
		}
		case 3:
		{
			MatIterator_<Vec3b> it, end;
			for (it = img_gamma.begin<Vec3b>(), end = img_gamma.end<Vec3b>(); it != end; it++)
			{

				(*it)[0] = lut[((*it)[0])];
				(*it)[1] = lut[((*it)[1])];
				(*it)[2] = lut[((*it)[2])];
			}
			break;
		}
	}

	if (show)
	{
		string window_name("Gamma Correction");
		namedWindow(window_name, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
		resizeWindow(window_name, 700, 280);

		Mat combined;
		hconcat(img_gamma, img, combined);
		imshow(window_name, combined);
	}

	return std::make_shared<Mat>(img_gamma);
}


std::pair<std::shared_ptr<Mat>, bool> Enhancement::ComputeHistogram(Mat& gray_img, bool show)
{
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat hist;
	bool uniform = true; 
	bool accumulate = false;
	calcHist(&gray_img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);


	int low_sum = 0;
	int high_sum = 0;
	bool ismostWhite = false;
	int thresh = histSize - roundf(histSize / 2.5);
	
	int index = 0;
	for (auto& it=hist.begin<uchar>(); it != hist.end<uchar>(); it++)
	{
		int pix_val = static_cast<int>(*it);
		if (index < thresh)
			low_sum += pix_val;
		else
			high_sum += pix_val;
	}
	
	if (high_sum > low_sum)
	{
		ismostWhite = true;
	}

	auto hist_ptr = std::make_shared<Mat>(hist);
	return std::make_pair(hist_ptr, ismostWhite);
}

std::shared_ptr<Mat> Enhancement::GaussianFilter(Mat& img_data, bool show = false)
{
	return make_shared<Mat>(img_data);

	Mat mat_gauss;
	auto pix_ptr = MatToPIX(img_data);
	auto pix_gauss = pixAddGaussianNoise(pix_ptr.get(), 0);
	return PIXToMat(std::make_shared<PIX>(*pix_ptr.get()));
}






