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

// using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000



int main(int argc, char **argv)
{

  bool use_dino2 = true;
  
  // FIXME: some changes to make it easier to edit paths, as we are using our own images: (make sure to change the path...)
  string groundtruth_path = "/workspaces/mono-vo/GT_FAST/01.txt";


  // string dataset_path  = "/workspaces/HyperImages/teagarden/session_000_001k/";
  // string dataset_path = "/workspaces/HyperImages/cornfields/session_002/";
  string dataset_path = "/workspaces/HyperTools/submodules/mono-vo/creek_2/";

  // string dataset_path = "/workspaces/HyperImages/wextel-1/";

  // calibration parameters for camera
  double focal = 718.8560;
  cv::Point2d pp(607.1928, 185.2157);

  // IMP: Change the file directories (4 places) according to where your dataset is saved before running!

  // if(use_dino2)
  // {
    // Load the TorchScript model
    torch::jit::script::Module module;
  
    try {
        module = torch::jit::load("/workspaces/HyperTools/dino_vitb16_model_test.pt");
        cout<<"Model loaded successfully!"<<endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }
    c10::intrusive_ptr<torch::ivalue::Tuple> output1, output2;
  // }

  cv::Mat img_1, img_2;
  cv::Mat R_f, t_f; // the final rotation and tranlation vectors containing the

  ofstream myfile;
  myfile.open("results1_1.txt"); // open up predicted

  ofstream ground_truth;
  ground_truth.open("ground_truth.txt");

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
  bool is_cu3 = false;
  bool is_png = false;
  while ((dp = readdir(dir)) != NULL)
  {
    if (strstr(dp->d_name, ".cu3") != NULL)
    {
      is_cu3 = true;
      break;
    }
    else if (strstr(dp->d_name, ".png") != NULL)
    {
      is_png = true;
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
    else if (strstr(dp->d_name, ".png") != NULL)
    {
      break;
    }
  }
  strcpy(filename2, dp->d_name);

  char text[100];
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);

  // set up hyperfunctions
  HyperFunctionsCuvis HyperFunctions1;
  cv::Mat img_1_c;
  cv::Mat img_2_c;

  if(is_cu3){
  // set up hyperfunctions
  HyperFunctions1.cubert_img = dataset_path + filename1;
  HyperFunctions1.dark_img = "/workspaces/HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3";
  HyperFunctions1.white_img = "/workspaces/HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3";
  HyperFunctions1.dist_img = "/workspaces/HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3";

  // generate false color image for first two images, convert to mat, layer values are for x20p
  HyperFunctions1.ReprocessImage(HyperFunctions1.cubert_img);
  HyperFunctions1.false_img_b = 25;
  HyperFunctions1.false_img_g = 40;
  HyperFunctions1.false_img_r = 78;
  HyperFunctions1.GenerateFalseImg();

  img_1_c = HyperFunctions1.false_img.clone();

  HyperFunctions1.cubert_img = dataset_path + filename2;
  HyperFunctions1.ReprocessImage(HyperFunctions1.cubert_img);
  HyperFunctions1.GenerateFalseImg();

  img_2_c = HyperFunctions1.false_img.clone();
  }
  else if(is_png){
    img_1_c = cv::imread((dataset_path + filename1).c_str());
    img_2_c = cv::imread((dataset_path + filename2).c_str());
  }

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

  vector<cv::Point2f> points1, points2; // vectors to store the coordinates of the feature points


  if(!use_dino2)
  {
    // convert to grayscale images
    cv::cvtColor(img_1_c, img_1_c, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2_c, img_2_c, cv::COLOR_BGR2GRAY);
    img_1 = img_1_c;
    img_2 = img_2_c;

    // feature detection, tracking
    
    featureDetection(img_1, points1); // detect features in img_1
    // AGASTDetection(img_1, points1);

  }
  else
  {
    // run model on image
    output1 = run_model_on_image(img_1_c, module);
    output2 = run_model_on_image(img_2_c, module);      
      
    cv::Mat features1 = tensorToMat(output1->elements()[0].toTensor());
    cv::Mat features2 = tensorToMat(output2->elements()[0].toTensor());

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    int grid_size = 14;

    int resize_scale = 224;

    // Convert features to keypoints
    for (int i = 0; i < (resize_scale/14); i++) {
        for(int j = 0; j < (resize_scale/14); j++) {
            int row, col;

            row = i * grid_size + grid_size / 2;
            col = j * grid_size + grid_size / 2;
            
            keypoints1.push_back(cv::KeyPoint(cv::Point2f(col, row), 1));
        }
    }

    keypoints2 = keypoints1;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    //  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    matcher->knnMatch( features1, features2, knn_matches, 2 );

    const float ratio_thresh = 0.9f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
    good_matches.push_back(knn_matches[i][0]);
    }
    }

    cout<<"good matches size "<<good_matches.size()<<endl;
 
    std::vector<cv::KeyPoint> matched_keypoints1, matched_keypoints2;


    // need to rescale points1 and points2' the points are in the range of 0-224, need to be rescaled to original image dimensions

    float rescale_amount_x = float(img_1_c.cols) / 224.0;
    float rescale_amount_y = float(img_1_c.rows ) / 224.0 ;
    // std::vector<cv::KeyPoint> keypoints1_rescaled, keypoints2_rescaled;


    // cout<<img_1_c.cols<<" rows " <<img_1_c.rows<<endl;
    for (const auto& match : good_matches) {
        matched_keypoints1.push_back(keypoints1[match.queryIdx]);
        matched_keypoints2.push_back(keypoints2[match.trainIdx]);

        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);

        cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
        pt1.x *= rescale_amount_x;
        pt1.y *= rescale_amount_y;
        pt1.x = std::floor(static_cast<int>(pt1.x));
        pt1.y = std::floor(static_cast<int>(pt1.y));
        points1.push_back(pt1);

        cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
        pt2.x *= rescale_amount_x;
        pt2.y *= rescale_amount_y;

        pt2.x = std::floor(static_cast<int>(pt2.x));
        pt2.y = std::floor(static_cast<int>(pt2.y));
        points2.push_back(pt2);

        // cout<<pt1<<" "<<pt2<<endl;
    }

    // for (const auto& pt : points1) {
    //     keypoints1_rescaled.push_back(cv::KeyPoint(pt, 1));
    // }

    // for (const auto& pt : points2) {
    //     keypoints2_rescaled.push_back(cv::KeyPoint(pt, 1));
    // }

    // std::vector<cv::DMatch> rescaled_matches;
    // for (size_t i = 0; i < good_matches.size(); i++) {
    //     cv::DMatch match = good_matches[i];
    //     match.queryIdx = i;
    //     match.trainIdx = i;
    //     rescaled_matches.push_back(match);
    // }
/*

  cv::Mat img1_resize , img2_resize ;
  cv::resize(img_1_c, img1_resize, cv::Size(224, 224), cv::INTER_LINEAR);
  cv::resize(img_2_c, img2_resize, cv::Size(224, 224), cv::INTER_LINEAR);
    cout<<"done here"<<endl;
     //-- Draw matches
 cv::Mat img_matches;
 // get keypoints from grid
 cv::drawMatches( img1_resize, keypoints1, img2_resize, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//  -- Show detected matches
 cv::imshow("Good Matches", img_matches );
 cv::waitKey(100);*/


  }
  vector<uchar> status;
  if (!use_dino2)
  {
    featureTracking(img_1, img_2, points1, points2, status); // track those features to img_2
  
  }
  else
  {
       cv::Mat img1_resize , img2_resize ;
       
      cv::resize(img_1_c, img1_resize, cv::Size(224, 224), cv::INTER_LINEAR);
      cv::resize(img_2_c, img2_resize, cv::Size(224, 224), cv::INTER_LINEAR);
      featureTracking(img1_resize, img2_resize, points1, points2, status); // track those features to img_2
  }
  
  cout<<"track 1"<<endl;

  // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
  
  // recovering the pose and the essential matrix
  cv::Mat E, R, t, mask;
  E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
  cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);

  cv::Mat prevImage = img_2;
  cv::Mat prevImage_c = img_2_c;
  cv::Mat currImage;
  vector<cv::Point2f> prevFeatures = points2;
  vector<cv::Point2f> currFeatures;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  cv::namedWindow("Road facing camera", cv::WINDOW_AUTOSIZE); // Create a window for display.
  cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);         // Create a window for display.

  cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

  while ((dp = readdir(dir)) != NULL)
  {
    cv::Mat currImage_c;
    if (strstr(dp->d_name, ".cu3") != NULL || strstr(dp->d_name, ".png") != NULL){
      if(is_cu3){
        strcpy(filename, dp->d_name);
        HyperFunctions1.cubert_img = dataset_path + filename;
        HyperFunctions1.ReprocessImage(HyperFunctions1.cubert_img);
        HyperFunctions1.GenerateFalseImg();
        currImage_c = HyperFunctions1.false_img.clone();
      }
      else if(is_png){
        strcpy(filename, (dataset_path+dp->d_name).c_str());
        currImage_c = cv::imread(filename);
      }
      // check if current image is empty
      if (currImage_c.empty())
      {
        std::cout << "Error: curr img is empty." << std::endl;
        return -1;
      }

      
      // convert to grayscale
      cv::cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY);
      vector<uchar> status;


      if(!use_dino2)
      {
      featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
      }
      else
      {

        featureTracking(prevImage_c, currImage_c, prevFeatures, currFeatures, status);
      }
      cout << currFeatures.size() << " features prev and cur " << prevFeatures.size() << endl;

      
      
      // redetect if images have less than 5 features
      if (currFeatures.size() < 5 || prevFeatures.size() < 5)
      {
        
        if(!use_dino2)
        {
          featureDetection(prevImage, prevFeatures);
          featureDetection(currImage, currFeatures);

          featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
        else
        {
          // run model on image
          cout<<"running model on image..."<<endl;
          output1 = run_model_on_image(prevImage_c, module);
          output2 = run_model_on_image(currImage_c, module);

          cv::Mat features1 = tensorToMat(output1->elements()[0].toTensor());

          cv::Mat features2 = tensorToMat(output2->elements()[0].toTensor());

          std::vector<cv::KeyPoint> keypoints1, keypoints2;
          int grid_size = 14;

          int resize_scale = 224;

          // Convert features to keypoints
          for (int i = 0; i < (resize_scale/14); i++) {
              for(int j = 0; j < (resize_scale/14); j++) {
                  int row, col;

                  row = i * grid_size + grid_size / 2;
                  col = j * grid_size + grid_size / 2;
                  
                  keypoints1.push_back(cv::KeyPoint(cv::Point2f(col, row), 1));
              }
          }

          keypoints2 = keypoints1;

          cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
          std::vector< std::vector<cv::DMatch> > knn_matches;
          //  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
          matcher->knnMatch( features1, features2, knn_matches, 2 );

          const float ratio_thresh = 0.9f;
          std::vector<cv::DMatch> good_matches;
          for (size_t i = 0; i < knn_matches.size(); i++)
          {
          if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
          {
          good_matches.push_back(knn_matches[i][0]);
          }
          }

          cout<<"good matches size place 2: "<<good_matches.size()<<endl;
      
          std::vector<cv::KeyPoint> matched_keypoints1, matched_keypoints2;

          for (const auto& match : good_matches) {
              matched_keypoints1.push_back(keypoints1[match.queryIdx]);
              matched_keypoints2.push_back(keypoints2[match.trainIdx]);

              points1.push_back(keypoints1[match.queryIdx].pt);
              points2.push_back(keypoints2[match.trainIdx].pt);
          }

          prevFeatures = points1;
          currFeatures = points2;

          featureTracking(prevImage_c, currImage_c, prevFeatures, currFeatures, status);

        }
        
        
        
        
        cout << "(!)----" << currFeatures.size() << " " << prevFeatures.size() << endl;
      }

      E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
      cout << E.rows << " " << E.cols << endl;

      // redetect if E is not 3x3
      if (E.rows != 3 || E.cols != 3)
      {
        cout<<"E is not 3x3"<<endl;
        
        
        if (!use_dino2)
        {
          // cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
          // cout << "trigerring redection" << endl;
          featureDetection(prevImage, prevFeatures);
          // AGASTDetection(prevImage, prevFeatures);
          featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        
        }
        else
        {

          // c10::intrusive_ptr<torch::ivalue::Tuple> output1, output2;
          // run model on image
          cout<<"running model on image..."<<endl;
          output1 = run_model_on_image(prevImage_c, module);
          output2 = run_model_on_image(currImage_c, module);

          cv::Mat features1 = tensorToMat(output1->elements()[0].toTensor());

          cv::Mat features2 = tensorToMat(output2->elements()[0].toTensor());

          std::vector<cv::KeyPoint> keypoints1, keypoints2;
          int grid_size = 14;

          int resize_scale = 224;

          // Convert features to keypoints
          for (int i = 0; i < (resize_scale/14); i++) {
              for(int j = 0; j < (resize_scale/14); j++) {
                  int row, col;

                  row = i * grid_size + grid_size / 2;
                  col = j * grid_size + grid_size / 2;
                  
                  keypoints1.push_back(cv::KeyPoint(cv::Point2f(col, row), 1));
              }
          }

          keypoints2 = keypoints1;

          cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
          std::vector< std::vector<cv::DMatch> > knn_matches;
          //  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
          matcher->knnMatch( features1, features2, knn_matches, 2 );

          const float ratio_thresh = 0.9f;
          std::vector<cv::DMatch> good_matches;
          for (size_t i = 0; i < knn_matches.size(); i++)
          {
          if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
          {
          good_matches.push_back(knn_matches[i][0]);
          }
          }

          cout<<"good matches size place 2: "<<good_matches.size()<<endl;
      
          std::vector<cv::KeyPoint> matched_keypoints1, matched_keypoints2;

          for (const auto& match : good_matches) {
              matched_keypoints1.push_back(keypoints1[match.queryIdx]);
              matched_keypoints2.push_back(keypoints2[match.trainIdx]);

              points1.push_back(keypoints1[match.queryIdx].pt);
              points2.push_back(keypoints2[match.trainIdx].pt);
          }

          prevFeatures = points1;
          currFeatures = points2;

          featureTracking(prevImage_c, currImage_c, prevFeatures, currFeatures, status);

        }

        // set prev image to current image and continue to next iteration
        prevImage = currImage.clone();
        prevImage_c = currImage_c.clone();
        prevFeatures = currFeatures;
        continue;

      }

      recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

      cv::Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

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
      if(is_cu3){
        char* const measurementLoc =  const_cast<char*>(HyperFunctions1.cubert_img.c_str());

        cuvis::Measurement mesu(measurementLoc);
      
        const cuvis::Measurement::gps_data_t* gps_data = mesu.get_gps();
        // Iterate over the GPS data and print the coordinates
        for (const auto& pair : *gps_data) {
            // std::cout << "Key: " << pair.first << ", Latitude: " << pair.second.latitude << ", Longitude: " << pair.second.longitude << std::endl;
            ground_truth << pair.second.latitude << " " << pair.second.longitude << endl;
        }
      }
      // lines for printing results
      myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

      // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
      if (prevFeatures.size() < MIN_NUM_FEAT && !use_dino2)
      {
        cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
        // cout << "trigerring redection" << endl;
        featureDetection(prevImage, prevFeatures);
        // AGASTDetection(prevImage, prevFeatures);

        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
      }

      prevImage = currImage.clone();
      prevImage_c = currImage_c.clone();
      prevFeatures = currFeatures;

      int x = int(t_f.at<double>(0)) + 300;
      int y = int(t_f.at<double>(2)) + 100;
      circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

      rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
      // display label on trajectory

      sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
      putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

      imshow("Road facing camera", currImage_c);
      imshow("Trajectory", traj);

      cv::waitKey(1);
    }
  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total CPU time taken: " << elapsed_secs << "s" << endl;

  // cout << R_f << endl;
  // cout << t_f << endl;

  return 0;
}