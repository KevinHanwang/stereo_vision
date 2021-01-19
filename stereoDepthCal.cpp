/******************************/
/*        立体匹配和测距        */
/******************************/

#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
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

int blockSize = 0, uniquenessRatio =0, numDisparities=0;
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

int ORBmatcher_high = 100;
int ORBmatcher_low  = 50;

int N ; //Number of Left KeyPoints.

float mb = 0.119; // Stereo baseline in meters.
float mbf = mb*349.0151; // Stereo baseline multiplied by fx.

// /*****立体匹配*****/
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

// /*****描述：鼠标操作回调*****/
// static void onMouse(int event, int x, int y, int, void*)
// {
//     if (selectObject)
//     {
//         selection.x = MIN(x, origin.x);
//         selection.y = MIN(y, origin.y);
//         selection.width = std::abs(x - origin.x);
//         selection.height = std::abs(y - origin.y);
//     }
//     switch (event)
//     {
//     case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
//         origin = Point(x, y);
//         selection = Rect(x, y, 0, 0);
//         selectObject = true;
//         cout << origin <<"in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
//         break;
//     case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
//         selectObject = false;
//         if (selection.width > 0 && selection.height > 0)
//         break;
//     }
// }

// 计算匹配点il和待匹配点ic的相似度dist, Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

/*****主函数*****/
int main(int argc, char** argv)
{
    /*
    立体校正
    */
    // Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        -1, imageSize, &validROIL, &validROIR);
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
    Mat _rgbRectifyImageL, _rgbRectifyImageR;
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, _rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, _rgbRectifyImageR, CV_GRAY2BGR);
    rgbRectifyImageL = _rgbRectifyImageL(Rect(120, 80, imageWidth - 240, imageHeight - 160));
    rgbRectifyImageR = _rgbRectifyImageR(Rect(120, 80, imageWidth - 240, imageHeight - 160));

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
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity",&blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    stereo_match(0,0);
    */


    // 特征提取与匹配
    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(rgbRectifyImageL, keypoints_1);
    detector->detect(rgbRectifyImageR, keypoints_2);
    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(rgbRectifyImageL, keypoints_1, descriptors_1);
    descriptor->compute(rgbRectifyImageR, keypoints_2, descriptors_2);
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                    [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(1.2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
        }
    }
    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(rgbRectifyImageL, keypoints_1, rgbRectifyImageR, keypoints_2, matches, img_match);
    drawMatches(rgbRectifyImageL, keypoints_1, rgbRectifyImageR, keypoints_2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //双目匹配
    N = keypoints_1.size();
    vector<float> mvuRight = vector<float>(N,-1.0f); //存储右图匹配点索引
    vector<float> mvDepth = vector<float>(N,-1.0f);  //存储特征点深度信息

    const int thOrbDist = (ORBmatcher_high+ORBmatcher_low)/2;

    const int nRows = rgbRectifyImageL.rows;

    //Step 1. 行特征点统计，存到vRowIndices中
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);
    const int Nr = keypoints_2.size();
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = keypoints_2[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f;
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Step 2 -> 3. 粗匹配 + 精匹配
    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);
    // 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = keypoints_1[iL];
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher_high;
        size_t bestIdxR = 0;
        const cv::Mat &dL = descriptors_1.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = keypoints_2[iR];

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = descriptors_2.row(iR);
                const int dist = DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Step 3. 精确匹配. Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            const float uR0 = keypoints_2[bestIdxR].pt.x;

            // sliding window search
            const int w = 5; // w表示sad相似度的窗口半径
            float Ro = rgbRectifyImageL.rows;
            float Co = rgbRectifyImageL.cols;
            // int rl = max(0.0f,kpL.pt.y-w);
            // int rh = min(kpL.pt.y+w+1, Ro);
            // int cl = max(0.0f, kpL.pt.x-w);
            // int ch = min(kpL.pt.x+w+1, Co);
            int rl = kpL.pt.y-w;
            int rh = kpL.pt.y+w+1;
            int cl = kpL.pt.x-w;
            int ch = kpL.pt.x+w+1;
            cv::Mat IL = rgbRectifyImageL.rowRange(rl, rh).colRange(cl,ch);
            IL.convertTo(IL,CV_32F);// convertTo()函数负责转换数据类型不同的Mat，即可以将类似float型的Mat转换到imwrite()函数能够接受的类型。
            // IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);
            IL = IL - IL.at<float>(w,w); //* Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = uR0+L-w;
            const float endu = uR0+L+w+1;
            if(iniu<0 || endu >= rgbRectifyImageR.cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = rgbRectifyImageR.rowRange(rl, rh).colRange(cl+incR,ch+incR);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w);// *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;

            float bestuR = (float)uR0+(float)bestincR+deltaR;
            float disparity = (uL-bestuR);
            if(disparity>=minD && disparity<maxD)
            {
                // 如果存在负视差，则约束为0.01
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    // Step 6. 删除离缺点(outliers)
    // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;
    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "matching cost = " << time_used.count() << " seconds. " << endl;

    // Mat outimg1;
    // drawKeypoints(rgbRectifyImageL, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    vector<float> mvDepCP;
    for(int iL=0; iL<N; iL++,iL++)
    {
        float _depth = mvDepth[iL];
        if(_depth == -1) continue;
        mvDepCP.push_back(_depth);
    }
    sort(mvDepCP.begin(), mvDepCP.end());
    int _size = mvDepCP.size();
    float mean = mvDepCP[_size/2];
    cout << "mean = " << mean << endl;

    int cnt = 0;
    for(int iL=0; iL<N; iL++,iL++)
    {
        const cv::KeyPoint &kpL = keypoints_1[iL];
        float _depth = mvDepth[iL];

        if(_depth == -1) continue;
        if(_depth > mean+0.01 || _depth < mean - 0.01) continue;

        _depth *= 0.8;
        cout << _depth << endl;
        cnt += 1;
        cv::Point p = cv::Point(kpL.pt.x + 120,kpL.pt.y+80+10*cnt);
        string s = to_string(_depth);
        Point point;
        point.x = kpL.pt.x+120;
        point.y = kpL.pt.y+80;
        cv::circle(_rgbRectifyImageL, point, 2, Scalar(0,0,255));
        cv::putText(_rgbRectifyImageL, s, p, cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 200, 200), 0.5, CV_AA);
    }
    namedWindow("Result", 0);
	cvResizeWindow("Result", 1800, 1200);
    imshow("Result", _rgbRectifyImageL);


    waitKey(0);
    return 0;
}