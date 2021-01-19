#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat M, N, B;
    M = (Mat_<float>(3,3) << 1,2,3,4,5,6,7,8,9);
    cout << "M = " << endl << " " << M << endl << endl;

    N = M - M.at<float>(2,2);
    cout << "N = " << endl << " " << N << endl << endl;

    B = M - M.at<float>(2,2) * Mat::ones(3,3, CV_32F);
    cout << "B = " << endl << " " << B << endl << endl;

    return 0;
}