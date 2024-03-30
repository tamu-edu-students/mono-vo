/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

//FIXME: THIS TESTS TO SEE IF WE HAVE THE INCLUDE PATH WORKING
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "../../../src/hyperfunctions.cpp"
#include "../../../src/hyperfunctions.h"
#include "../../../src/hypercuvisfunctions.h"
#include "../../../src/hypercuvisfunctions.cpp"

#include <dirent.h>


#include <torch/torch.h>
#include <torch/script.h> 
#include <memory>
#include <cmath>

// #include "opencv2/video/tracking.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/features2d/features2d.hpp"
// #include "opencv2/calib3d/calib3d.hpp"
// #include "opencv2/calib3d/calib3d.hpp"



#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

// using namespace cv;
using namespace std;

void featureTracking(cv::Mat img_1, cv::Mat img_2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  cv::Size winSize=cv::Size(21,21);																								
  cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}


void featureDetection(cv::Mat img_1, vector<cv::Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  vector<cv::KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints_1, points1, vector<int>());
}

void AGASTDetection(cv::Mat img_1, vector<cv::Point2f>& points1)	{   //uses AGAST as of now, modify parameters as necessary
  vector<cv::KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  AGAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints_1, points1, vector<int>());
}


c10::intrusive_ptr<torch::ivalue::Tuple> run_model_on_image(const cv::Mat& input_img, torch::jit::script::Module& module) {
    cv::Mat img;
    input_img.copyTo(img);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_LINEAR);

    img.convertTo(img, CV_32F, 1.0 / 255);

    auto tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3});
    if (tensor.dim() == 4) {
        tensor = tensor.permute({0, 3, 1, 2});
    } else {
        std::cerr << "Tensor does not have 4 dimensions\n";
        return nullptr;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);

    auto output = module.forward(inputs);

    if (!output.isTuple()) {
        std::cerr << "Output is not a tuple\n";
        return nullptr;
    }

    auto output_tuple = output.toTuple();
    // std::cout << "Tuple size: " << output_tuple->elements().size() << std::endl;

    // for (size_t i = 0; i < output_tuple->elements().size(); ++i) {
    //     auto element = output_tuple->elements()[i];
    //     if (element.isTensor()) {
    //         // std::cout << "Element " << i << ": " << element.toTensor() << std::endl;
    //        torch::ScalarType dtype = element.toTensor().scalar_type();
    //         std::cout << "Element " << i << " data type: " << dtype << std::endl;
    //     }
    // }

    // Return the entire output tuple
    return output_tuple;
}


cv::Mat tensorToMat(torch::Tensor tensor, bool is_gray=false) {
    tensor = tensor.detach().cpu();
    tensor = tensor.squeeze();
    if (tensor.dim() == 3) {
        tensor = tensor.permute({1, 2, 0});
    } else if (tensor.dim() != 2) {
        std::cerr << "Tensor does not have 2 or 3 dimensions\n";
        return cv::Mat();
    }
    tensor = tensor.to(torch::kF32);
    // cv::Mat mat(cv::Size{tensor.size(1), tensor.size(0)}, CV_32FC1);
    cv::Mat mat(cv::Size{static_cast<int>(tensor.size(1)), static_cast<int>(tensor.size(0))}, CV_32F);
    std::memcpy((void*)mat.data, tensor.data_ptr(), sizeof(float)*tensor.numel());
    return mat.clone();
}


double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)
{

  string line;
  int i = 0;
  ifstream myfile(groundtruth_path);
  double x = 0, y = 0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while ((getline(myfile, line)) && (i <= frame_id)) // map out the scale of ground truth
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      // cout << line << '\n';
      for (int j = 0; j < 12; j++)
      {
        in >> z;
        if (j == 7)
          y = z;
        if (j == 3)
          x = z;
      }

      i++;
    }
    myfile.close();
  }

  else
  {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));
}