/******************************/
/*        立体匹配和测距        */
/******************************/

#include <opencv2/opencv.hpp>  
#include <iostream>
#include <chrono>  

using namespace std;
using namespace cv;

const int imageWidth = 672;                             //摄像头的分辨率  
const int imageHeight = 376;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 8, uniquenessRatio =23, numDisparities=7;
Ptr<StereoBM> bm = StereoBM::create(16, 9);

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 349.0151, 0.0525, 343.1174,
    0, 349.1126, 185.3036,
    0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.1051, -0.0260, 0.0, -0.000219, 0.0381);

Mat cameraMatrixR = (Mat_<double>(3, 3) <<  350.3199, 0.2725, 345.0589,
    0, 350.0103, 182.3179,
    0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.1660, 0.0142, -0.000255, 0.0, 0.0073);

Mat T = (Mat_<double>(3, 1) << -119.8684, -0.0184, -0.6295);//T平移向量
// Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
Mat R = (Mat_<double>(3, 3) <<  1.0000, -0.0005787, -0.0093,
    0.0006, 1.0000, 0.0027,
    0.0093, -0.0027, 1.0000);//R 旋转矩阵


/*****立体匹配*****/
// void stereo_match(int,void*)
// {
//     bm->setBlockSize(2*blockSize+5);     //SAD窗口大小，5~21之间为宜
//     bm->setROI1(validROIL);
//     bm->setROI2(validROIR);
//     bm->setPreFilterCap(31);
//     bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
//     bm->setNumDisparities(numDisparities*16+16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
//     bm->setTextureThreshold(10); 
//     bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
//     bm->setSpeckleWindowSize(100);
//     bm->setSpeckleRange(32);
//     bm->setDisp12MaxDiff(-1);
//     Mat disp, disp8;
//     bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
//     disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
//     reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
//     xyz = xyz * 16;
//     imshow("disparity", disp8);
// }

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin <<"in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
        break;
    }
}


/*****主函数*****/
int main(int argc, char** argv)
{
    /*
    立体校正
    */
    // Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    读取图片
    */
    Mat frame = imread(argv[1]);
    Mat leftImage, rightImage;
    rgbImageL = frame(Rect(0, 0, frame.size().width / 2, frame.size().height));//split left image
    rgbImageR = frame(Rect(frame.size().width / 2, 0, frame.size().width / 2, frame.size().height));//split right image
    cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
    cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

    imshow("ImageL Before Rectify", grayImageL);
    imshow("ImageR Before Rectify", grayImageR);

    /*
    经过remap之后，左右相机的图像已经共面并且行对准了
    */
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    /*
    把校正结果显示出来
    */
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    imshow("ImageL After Rectify", rgbRectifyImageL);
    imshow("ImageR After Rectify", rgbRectifyImageR);

    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);


    /*
    立体匹配
    */
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);

    bm->setBlockSize(2*blockSize+5);     //SAD窗口大小，5~21之间为宜
    bm->setROI1(validROIL);
    bm->setROI2(validROIR);
    bm->setPreFilterCap(31);
    bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
    bm->setNumDisparities(numDisparities*16+16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    bm->setTextureThreshold(10); 
    bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);
    Mat disp, disp8;
    bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    imshow("disparity", disp8);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "matching cost = " << time_used.count() << " seconds. " << endl;

    // // 创建SAD窗口 Trackbar
    // createTrackbar("BlockSize:\n", "disparity",&blockSize, 8, stereo_match);
    // // 创建视差唯一性百分比窗口 Trackbar
    // createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // // 创建视差窗口 Trackbar
    // createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    // stereo_match(0,0);

    waitKey(0);
    return 0;
}