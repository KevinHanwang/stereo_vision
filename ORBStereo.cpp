/*
 * 双目匹配函数
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n ‘
 * 这里所说的SAD是一种双目立体视觉匹配算法，可参考[https://blog.csdn.net/u012507022/article/details/51446891]
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n 
 * 这里所谓的亚像素精度，就是使用这个拟合得到一个小于一个单位像素的修正量，这样可以取得更好的估计结果，计算出来的点的深度也就越准确
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
    /*两帧图像稀疏立体匹配（即：ORB特征点匹配，非逐像素的密集匹配，但依然满足行对齐）
     * 输入：两帧立体矫正后的图像img_left 和 img_right 对应的orb特征点集
     * 过程：
          1. 行特征点统计. 统计img_right每一行上的ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断.
          2. 粗匹配. 根据步骤1的结果，对img_left第i行的orb特征点pi，在img_right的第i行上的orb特征点集中搜索相似orb特征点, 得到qi
          3. 精确匹配. 以点qi为中心，半径为r的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
          4. 亚像素精度优化. 步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素差值（抛物线插值)获取float精度的真实视差
          5. 最优视差值/深度选择. 通过胜者为王算法（WTA）获取最佳匹配点。
          6. 删除离缺点(outliers). 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
     * 输出：稀疏特征点视差图/深度图（亚像素精度）mvDepth 匹配结果 mvuRight
     */

    // 为匹配结果预先分配内存，数据类型为float型
    // mvuRight存储右图匹配点索引
    // mvDepth存储特征点的深度信息
	mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

	// orb特征相似度阈值  -> mean ～= (max  + min) / 2 =75;
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    // 金字塔顶层（0层）图像高 nRows
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

	// 二维vector存储每一行的orb特征点的列坐标，为什么是vector，因为每一行的特征点有可能不一样，例如
    // vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
    // vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
    vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());
    for(int i=0; i<nRows; i++） vRowIndices[i].reserve(200);

	// 右图特征点数量，N表示数量 r表示右图，且不能被修改
    const int Nr = mvKeysRight.size();

	// Step 1. 行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    for(int iR = 0; iR < Nr; iR++) {

        // 获取特征点ir的y坐标，即行号
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        
        // 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY -r]
        // 2 表示在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        // r= ２＊１、２＊0.83^1、２*0.83^2...
        const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        // 将特征点ir保证在可能的行号中
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Step 2 -> 3. 粗匹配 + 精匹配　
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视差mind
    // 也即是左图中任何一点p，在右图上的匹配点的范围为应该是[p - maxd, p - mind], 而不需要遍历每一行所有的像素
    // maxd = baseline * length_focal / minZ
    // mind = baseline * length_focal / maxZ

    // Stereo baseline multiplied by fx.
    ///float mbf; baseline x fx
    

    // Stereo baseline in meters.
    // float mb; 相机的基线长度,单位为米 ,mb = mbf/fx;(z=f*b/d:d视差，ｚ深度）
    
    const float minZ = mb;//minZ深度
    const float minD = 0;			 
    const float maxD = mbf/minZ; //mbf:相机的基线长度 * 相机的焦距

    // 保存sad块匹配相似度和左图特征点索引
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for(int iL=0; iL<N; iL++) {

		const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];
        if(vCandidates.empty()) continue;

        // 计算理论上的最佳搜索范围
        const float minU = uL-maxD;//改行中搜索最小列
        const float maxU = uL-minD;//改行中搜索最大列，就是其ｖｌ=uL=ｍａｘU
        
        // 最大搜索范围小于0，说明无匹配点
        if(maxU<0) continue;

		// 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        int bestDist = ORBmatcher::TH_HIGH;//100;
        size_t bestIdxR = 0;
        //(il:作图特征点编号)左目摄像头和右目摄像头特征点对应的描述子 mDescriptors, mDescriptorsRight;
        const cv::Mat &dL = mDescriptors.row(iL);//dL用来计算描述子的汉明距离；但描述子的row表示什么？
        
        // Step2. 粗配准. 左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的相似度和索引
        for(size_t iC=0; iC<vCandidates.size(); iC++) {

            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 左图特征点il与带匹配点ic的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;

            // 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if(uR >= minU && uR <= maxU) {

                // 计算匹配点il和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

				//统计最小相似度及其对应的列坐标(x)
                //初始bestDist=??
                //初始bestIdxR=0
                if( dist<bestDist ) {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }
        
        // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
        // Step 3. 精确匹配. 
        // const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2=75;
        if(bestDist<thOrbDist) {
            // 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            
            // 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);			
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // 滑动窗口搜索, 类似模版卷积或滤波
            // w表示sad相似度的窗口半径
            const int w = 5;

            // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            // convertTo()函数负责转换数据类型不同的Mat，即可以将类似float型的Mat转换到imwrite()函数能够接受的类型。
            IL.convertTo(IL,CV_32F);
            
            // 图像块均值归一化，降低亮度变化对相似度计算的影响
			IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);

			//初始化最佳相似度
            int bestDist = INT_MAX;

			// 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;

			//滑动窗口的滑动范围为（-L, L）
            const int L = 5;

			// 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2*L+1); 

            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
            // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            const float iniu = scaleduR0+L-w;// scaleduR0:右图粗匹配到的金字塔尺度的特征点坐标ｘ
            const float endu = scaleduR0+L+w+1;

			// 判断搜索是否越界
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

			// 在搜索范围内从左到右滑动，并计算图像块相似度
            for(int incR=-L; incR<=+L; incR++) {

                // 提取右图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                
                // 图像块均值归一化，降低亮度变化对相似度计算的影响
                IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);
                
                // sad 计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);

                // 统计最小sad和偏移量
                if(dist<bestDist) {
                    bestDist = dist;
                    bestincR = incR;
                }

                //L+incR 为refine后的匹配点列坐标(x)
                // vector<float> vDists;vDists.resize(2*L+1)=11;
                vDists[L+incR] = dist; 	 
            }

            // 搜索窗口越界判断ß 
            if(bestincR==-L || bestincR==L)
                continue;

			// Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
            //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
            //      .             .(16)
            //         .14     .(15) <- int/uchar最佳视差值
            //              . 
            //           （14.5）<- 真实的视差值
            //   deltaR = 15.5 - 16 = -0.5
            // 公式参考opencv sgbm源码中的亚像素插值公式
            // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7

            const float dist1 = vDists[L+bestincR-1];	//bestincR:列坐标偏移量
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;
            
            // 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);
            // disparity:求得的视差值
            float disparity = (uL-bestuR);
            if(disparity>=minD && disparity<maxD) {
                // 如果存在负视差，则约束为0.01
                if( disparity <=0 ) {
                    disparity=0.01;// ?? 为啥约束0.01？
                    bestuR = uL-0.01;　//uL=kpL.pt.x,左图像x点的列数值
                }
                
                // 根据视差值计算深度信息
                // 保存最相似点的列坐标(x)信息
                // 保存归一化sad最小相似度
                // Step 5. 最优视差值/深度选择.
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;//根据亚像素精度偏移量delta调整最佳匹配索引,视差disparity = (uL-bestuR);
                // vDistIdx 二维数组存放 ＳＡＤ计算的最小滑块匹配值和左图这个特征的编号
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
        } //end if   
    }//遍历左图每个特征点

    // Step 6. 删除离缺点(outliers)
    // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;//阈值

    for(int i=vDistIdx.size()-1;i>=0;i--) {
        if(vDistIdx[i].first<thDist)
            break;
        else {
			// 误匹配点置为-1，和初始化时保持一直，作为error code
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }                                                                                                                                               
}
