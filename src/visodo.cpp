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

#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

// FIXME: some changes to make it easier to edit paths, as we are using our own images: (make sure to change the path...)
string groundtruth_path = "/workspaces/mono-vo/GT_FAST/01.txt";

// string dataset_path  = "/workspaces/HyperImages/teagarden/session_000_001k/";
string dataset_path = "/workspaces/HyperImages/cornfields/session_002/";

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

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

int main(int argc, char **argv)
{

  Mat img_1, img_2;
  Mat R_f, t_f; // the final rotation and tranlation vectors containing the

  ofstream myfile;
  myfile.open("results1_1.txt"); // open up predicted

  double scale = 1.00;
  DIR *dir;
  struct dirent *dp;

  if ((dir = opendir(dataset_path.c_str())) == NULL)
  {
    perror("Cannot open datasets.");
    exit(1);
  }

  char filename1[200];
  char filename2[200];

  // read the first two frames from the dataset
  // image 1
  dp = readdir(dir);
  while ((dp = readdir(dir)) != NULL)
  {
    if (strstr(dp->d_name, ".cu3") != NULL)
    {
      break;
    }
  }
  strcpy(filename1, dp->d_name);

  // image 2
  dp = readdir(dir);
  while ((dp = readdir(dir)) != NULL)
  {
    if (strstr(dp->d_name, ".cu3") != NULL)
    {
      break;
    }
  }
  strcpy(filename2, dp->d_name);

  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);

  // set up hyperfunctions
  HyperFunctionsCuvis HyperFunctions1;
  HyperFunctions1.cubert_img = dataset_path + filename1;
  HyperFunctions1.dark_img = "/workspaces/HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
  HyperFunctions1.white_img = "/workspaces/HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
  HyperFunctions1.dist_img = "/workspaces/HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";

  // generate false color image for first two images, convert to mat
  HyperFunctions1.ReprocessImage(HyperFunctions1.cubert_img);
  HyperFunctions1.false_img_b = 2;
  HyperFunctions1.false_img_g = 13;
  HyperFunctions1.false_img_r = 31;
  HyperFunctions1.GenerateFalseImg();

  Mat img_1_c = HyperFunctions1.false_img;

  HyperFunctions1.cubert_img = dataset_path + filename2;
  HyperFunctions1.ReprocessImage(HyperFunctions1.cubert_img);
  HyperFunctions1.GenerateFalseImg();

  Mat img_2_c = HyperFunctions1.false_img;

  // check if images are empty
  if (!img_1_c.data || !img_2_c.data)
  {
    std::cout << " --(!) Error reading images " << std::endl;
    return -1;
  }

  if (img_1_c.empty())
  {
    std::cout << "Error: img_1_c is empty." << std::endl;
    return -1;
  }

  if (img_2_c.empty())
  {
    std::cout << "Error: img_2_c is empty." << std::endl;
    return -1;
  }

  // convert to grayscale images
  cvtColor(img_1_c, img_1_c, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2_c, COLOR_BGR2GRAY);
  img_1 = img_1_c;
  img_2 = img_2_c;

  // feature detection, tracking
  vector<Point2f> points1, points2; // vectors to store the coordinates of the feature points
  featureDetection(img_1, points1); // detect features in img_1
  // AGASTDetection(img_1, points1);

  vector<uchar> status;
  featureTracking(img_1, img_2, points1, points2, status); // track those features to img_2

  // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
  double focal = 718.8560;
  cv::Point2d pp(607.1928, 185.2157);
  // recovering the pose and the essential matrix
  Mat E, R, t, mask;
  E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points2, points1, R, t, focal, pp, mask);

  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  namedWindow("Road facing camera", WINDOW_AUTOSIZE); // Create a window for display.
  namedWindow("Trajectory", WINDOW_AUTOSIZE);         // Create a window for display.

  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  while ((dp = readdir(dir)) != NULL)
  {
    if (strstr(dp->d_name, ".cu3") != NULL)
    {
      // generate false color image for current image, convert to mat
      strcpy(filename, dp->d_name);
      HyperFunctions1.cubert_img = dataset_path + filename;
      HyperFunctions1.ReprocessImage(HyperFunctions1.cubert_img);
      HyperFunctions1.GenerateFalseImg();
      Mat currImage_c = HyperFunctions1.false_img;

      // check if current image is empty
      if (currImage_c.empty())
      {
        std::cout << "Error: curr img is empty." << std::endl;
        return -1;
      }

      // convert to grayscale
      cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
      vector<uchar> status;

      featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
      // cout << currFeatures.size() << " " << prevFeatures.size() << endl;

      // redetect if images have less than 5 features
      if (currFeatures.size() < 5 || prevFeatures.size() < 5)
      {
        featureDetection(prevImage, prevFeatures);
        featureDetection(currImage, currFeatures);

        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        // cout << "(!)----" << currFeatures.size() << " " << prevFeatures.size() << endl;
      }

      E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
      // cout << E.rows << " " << E.cols << endl;

      // redetect if E is not 3x3
      if (E.rows != 3 || E.cols != 3)
      {
        // cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
        // cout << "trigerring redection" << endl;
        featureDetection(prevImage, prevFeatures);
        // AGASTDetection(prevImage, prevFeatures);
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        // set prev image to current image and continue to next iteration
        prevImage = currImage.clone();
        prevFeatures = currFeatures;
        continue;
      }

      recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

      Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

      for (int i = 0; i < prevFeatures.size(); i++)
      { // this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
        prevPts.at<double>(0, i) = prevFeatures.at(i).x;
        prevPts.at<double>(1, i) = prevFeatures.at(i).y;

        currPts.at<double>(0, i) = currFeatures.at(i).x;
        currPts.at<double>(1, i) = currFeatures.at(i).y;
      }

      // scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
      scale = 1;

      // cout << "Scale is " << scale << endl;

      if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
      {

        t_f = t_f + scale * (R_f * t);
        R_f = R * R_f;
      }

      else
      {
        // cout << "scale below 0.1, or incorrect translation" << endl;
      }

      // lines for printing results
      myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

      // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
      if (prevFeatures.size() < MIN_NUM_FEAT)
      {
        // cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
        // cout << "trigerring redection" << endl;
        featureDetection(prevImage, prevFeatures);
        // AGASTDetection(prevImage, prevFeatures);

        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
      }

      prevImage = currImage.clone();
      prevFeatures = currFeatures;

      int x = int(t_f.at<double>(0)) + 300;
      int y = int(t_f.at<double>(2)) + 100;
      circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

      rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), FILLED);
      // display label on trajectory

      sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
      putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

      imshow("Road facing camera", currImage_c);
      imshow("Trajectory", traj);

      waitKey(1);
    }
  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  // cout << R_f << endl;
  // cout << t_f << endl;

  return 0;
}