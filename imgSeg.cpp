#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
using namespace cv;
using namespace std;
int n = 27;
int main(int argc, char** argv)
{
    char file1[100];
    char file2[100];
    for(int i=1; i<=n; ++i){
        Mat frame = imread(argv[i]);
        Mat leftImage, rightImage;
        leftImage = frame(Rect(0, 0, frame.size().width / 2, frame.size().height));//split left image
        rightImage = frame(Rect(frame.size().width / 2, 0, frame.size().width / 2, frame.size().height));//split right image
        
        sprintf(file1, "/home/wanghan/stereo/build/left/l%u.jpg",i);
        sprintf(file2, "/home/wanghan/stereo/build/right/r%u.jpg",i);
        imwrite(file1, leftImage);
        imwrite(file2, rightImage);
    }
	
	return 0;
}
