#include <iostream>
#include <vector>
#include <fstream>
//
#include <opencv2/opencv.hpp>
//
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
//
#ifdef _OPENMP
    #include <omp.h>
#endif
//
const char *http200_ok = "HTTP/1.1 200 OK\
                         Content-Length: 0\
                         Content-Type: text/html";

const char *http404_not_found = "HTTP/1.1 404 Not found\
                         Content-Length: 0\
                         Content-Type: text/html";

//
bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

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
//
bool getBaseDirAndFileName(const std::string fullpath, std::string &baseDir, std::string &fileName)
{
    if(fullpath.find_last_of("/") + 1 > fullpath.length())
        return false;
    fileName = fullpath.substr(fullpath.find_last_of("/") + 1);
    baseDir = fullpath.substr(0, fullpath.find_last_of("/") + 1);
    return true;
}
//
bool getFileNameAndExtension(const std::string fileNameWithExtension, std::string &fileName, std::string &extension)
{
    auto dotPos = fileNameWithExtension.find(".");
    if( dotPos == std::string::npos)
        return false;
    fileName = fileNameWithExtension.substr(0, dotPos);
    extension = fileNameWithExtension.substr(dotPos + 1);
    return true;
}
bool createVideoCollage(const std::string sourceVideoPath, unsigned brightnessMultiplier)
{
    cv::VideoCapture videoFromFile(sourceVideoPath);
    if(!videoFromFile.isOpened())
    {
        std::cout << "Can not open video" << std::endl;
        return false;
    }

    int frameWidth = videoFromFile.get(cv::CAP_PROP_FRAME_WIDTH) * 3;
    int frameHeight = videoFromFile.get(cv::CAP_PROP_FRAME_HEIGHT);
    int originalVideoFrameRate = videoFromFile.get(cv::CAP_PROP_FPS);
    cv::Mat currentFrame;
    //
    std::string baseDir, fileName;
    if(!getBaseDirAndFileName(sourceVideoPath, baseDir, fileName))
        return false;
    //
    std::string fileNameWithoutExtension, fileExtension;
    if(!getFileNameAndExtension(fileName, fileNameWithoutExtension, fileExtension))
        return false;
    cv::VideoWriter outputVideo(baseDir + fileNameWithoutExtension +"_processed" + "." + fileExtension,
                                cv::VideoWriter::fourcc('M','J','P','G'),originalVideoFrameRate, cv::Size(frameWidth, frameHeight));
    if(!outputVideo.isOpened())
    {
        std::cout << "Can not open video output file" << std::endl;
        return false;
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
    return true;
}

//
void processCommandsFromNetwork()
{
        int sock, listener;
        struct sockaddr_in addr;
        char buf[1024];
        int bytes_read;

        listener = socket(AF_INET, SOCK_STREAM, 0);
        if(listener < 0)
        {
            std::cout << "Can not create socket!" << std::endl;
            return;
        }

        addr.sin_family = AF_INET;
        addr.sin_port = htons(8081);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if(bind(listener, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        {
            std::cout << "Can not bind to address!" << std::endl;
            exit(2);
        }

        listen(listener, 1);

        while(true)
        {
            sock = accept(listener, NULL, NULL);
            if(sock < 0)
            {
                std::cout << "Accept problem!" << std::endl;
                exit(3);
            }
            std::string postRequest;
            while(true)
            {
                bytes_read = recv(sock, buf, 1024, 0);
                if(bytes_read <= 0) break;
                postRequest += buf;
            }

            if(postRequest.find("\r\n") != std::string::npos)
            {
                auto paramsString = postRequest.substr(postRequest.find("\r\n") + 2);
                const std::string filePathTag = "input_video=";
                auto posOfFilePath = paramsString.find(filePathTag);
                if(posOfFilePath !=  std::string::npos)
                {
                    auto posOfDelimeter = paramsString.find("&");
                    if(posOfDelimeter != std::string::npos)
                    {
                        auto startOfPath = posOfFilePath + filePathTag.length();
                        auto fullFilePath = paramsString.substr(startOfPath,
                                                                posOfDelimeter - startOfPath);
                        std::cout << "filepath: " << fullFilePath << std::endl;
                        const std::string brightnessMultiplicatorTag = "brightness_multiplicator=";
                        auto posOfBrightnessMultiplicator = paramsString.find(brightnessMultiplicatorTag);
                        if(posOfBrightnessMultiplicator != std::string::npos)
                        {
                            auto startOfMultiplicator = posOfBrightnessMultiplicator +
                                                        brightnessMultiplicatorTag.length();
                            auto multiplicatorStringValue = paramsString.substr(startOfMultiplicator);
                            std::cout << "Multiplicator: " << multiplicatorStringValue << std::endl;
                            //все параметры получены, проверяем наличие файла
                            if(is_file_exist(fullFilePath.c_str()))//отправляем ОК 200 обратно
                            {
                                send(sock, http200_ok, strlen(http200_ok), 0);
                                //обрабатываем видео
                                createVideoCollage(fullFilePath, std::stoul(multiplicatorStringValue));
                            }
                            else
                            {
                                send(sock, http404_not_found, strlen(http404_not_found), 0);
                            }
                        }
                    }
                }
            }
            std::cout.flush();
            close(sock);
        }
}
//
int main()
{
       processCommandsFromNetwork();
       return 0;
}
