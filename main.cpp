#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef _OPENMP
    #include <omp.h>
#endif
cv::Mat createCollage(cv::Mat sourceImage, int brightnessMultiplier)
{
    if ( sourceImage.size().empty())
        return cv::Mat();

    std::vector<cv::Mat> bgr_planes;
    cv::split( sourceImage, bgr_planes );
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    //
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    //
    int hist_w = sourceImage.size().width, hist_h = sourceImage.size().height;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    //
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    //
    for( int i = 1; i < histSize; i++ )
    {
        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              cv::Scalar( 255, 0, 0), 2, 8, 0  );
        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              cv::Scalar( 0, 255, 0), 2, 8, 0  );
        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              cv::Scalar( 0, 0, 255), 2, 8, 0  );
    }
    //
    cv::Mat imageWithIncreasedBrightness = cv::Mat::zeros( sourceImage.size(), sourceImage.type() );
    //
    double alpha = 1.0; //множитель контрастности, который оставим 1
    int beta = 0; // множитель яркости, которую будем менять

    //находим среднее значение яркости среди всех пикселей
    unsigned long long sumOfAllPixels = 0;
    for( int i = 0; i < sourceImage.rows; i++ ) {
            for( int j = 0; j < sourceImage.cols; j++ ) {
                for( int k = 0; k < sourceImage.channels(); k++ ) {
                    sumOfAllPixels += sourceImage.at<cv::Vec3b>(i,j)[k];
                }
            }
        }
    unsigned long avgBrightness = sumOfAllPixels / (sourceImage.rows * sourceImage.cols * sourceImage.channels());
    //std::cout << "Sum of all pixels: " << sumOfAllPixels << "\nAverage brightness: " << avgBrightness << std::endl;
    beta = avgBrightness * ( brightnessMultiplier - 1 );
    //
    #pragma omp parallel for
    for( int y = 0; y < sourceImage.rows; y++ ) {
            for( int x = 0; x < sourceImage.cols; x++ ) {
                for( int c = 0; c < sourceImage.channels(); c++ ) {
                    imageWithIncreasedBrightness.at<cv::Vec3b>(y,x)[c] =
                      cv::saturate_cast<uchar>( alpha*sourceImage.at<cv::Vec3b>(y,x)[c] + beta );
                }
            }
        }
    //сшиваем 3 изображения
    unsigned numImagesInCollage = 3;
    cv::Mat resultedCollage = //cv::Mat::zeros( sourceImage.size(), sourceImage.type() );
            cv::Mat::zeros(sourceImage.rows , sourceImage.cols * numImagesInCollage, sourceImage.type());
    //исходное изображение
    sourceImage.copyTo(resultedCollage(cv::Rect(0, 0, sourceImage.size().width, sourceImage.size().height)));
    //Гистограмма по трем каналам
    histImage.copyTo(resultedCollage(cv::Rect(sourceImage.cols, 0, histImage.size().width, histImage.size().height)));
    //Изображение, на котором исходная яркость увеличена в Х раз
    imageWithIncreasedBrightness.copyTo(resultedCollage(cv::Rect(sourceImage.cols * 2, 0,
                   imageWithIncreasedBrightness.size().width, imageWithIncreasedBrightness.size().height)));
    return resultedCollage;
}
int main()
{
       unsigned brightnessMultiplier = 3; // тот самый множитель, который мы получаем извне
       cv::VideoCapture videoFromFile("/home/smak/video_sample.mp4");
       if(!videoFromFile.isOpened())
       {
           std::cout << "Can not open video" << std::endl;
           return 1;
       }

       int frameWidth = videoFromFile.get(cv::CAP_PROP_FRAME_WIDTH) * 3;
       int frameHeight = videoFromFile.get(cv::CAP_PROP_FRAME_HEIGHT);
       int originalVideoFrameRate = videoFromFile.get(cv::CAP_PROP_FPS);
       cv::Mat currentFrame;
       cv::VideoWriter outputVideo("/home/smak/collageVideo.avi",
                                   cv::VideoWriter::fourcc('M','J','P','G'),originalVideoFrameRate, cv::Size(frameWidth, frameHeight));
       if(!outputVideo.isOpened())
       {
           std::cout << "Can not open video output file" << std::endl;
           return 1;
       }
       while(true) // покадрово
       {
           videoFromFile >> currentFrame;
           if(currentFrame.empty())
               break;
           outputVideo << createCollage(currentFrame, brightnessMultiplier);
       }
       videoFromFile.release();
       outputVideo.release();
       cv::Mat sourceImage;
       sourceImage = cv::imread("/home/smak/test.jpg");



       auto resultedCollage = createCollage(sourceImage,brightnessMultiplier);
       cv::imwrite("/home/smak/collage.jpg", resultedCollage);
       cv::waitKey();
       return 0;
}
