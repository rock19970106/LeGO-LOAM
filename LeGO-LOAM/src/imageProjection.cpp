// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"

class ImageProjection {
private:

    ros::NodeHandle nh;//ROS句柄

    //一个发布者
    ros::Subscriber subLaserCloud;

    //多个发布者
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;//地面点云
    ros::Publisher pubSegmentedCloud;//只去除了outlier点的点云(降采样)
    ros::Publisher pubSegmentedCloudPure;//去除了outlier点和地面点的所有的点云(没有降采样)
    ros::Publisher pubSegmentedCloudInfo;//自定义消息类型
    ros::Publisher pubOutlierCloud;//不属于规则聚类的点

    //用于发布与处理的点云数据
    pcl::PointCloud<PointType>::Ptr laserCloudIn;//接受到的来自激光Msg的原始点云数据
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;//用 laserCloudInRing 存储含有具有通道R的原始点云数据

    //深度图点云：以一维形式存储与深度图像素一一对应的点云数据
    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    //带距离值的深度图点云:与深度图点云存储一致的数据，但是其属性intensity记录的是距离值
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    //注：所有点分为被分割点、未被分割点、地面点、无效点。
    pcl::PointCloud<PointType>::Ptr groundCloud;//地面点点云
    pcl::PointCloud<PointType>::Ptr segmentedCloud;//segMsg 点云数据:包含被分割点和经过降采样的地面点
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;//存储被分割点点云，且每个点的i值为分割标志值
    pcl::PointCloud<PointType>::Ptr outlierCloud;//经过降采样的未分割点

    PointType nanPoint; // fill in fullCloud at each iteration

    //由激光点云数据投影出来的深度图像
    //1） rangeMat.at(i,j) = FLT_MAX，浮点数的最大值，初始化信息；
    //2） rangeMat.at(rowIdn, columnIdn) = range，保存图像深度；
    cv::Mat rangeMat; // range matrix for range image

    //分割的标志矩阵：每一个数字代表一个分割的类别,大值表示未分割点,-1表示不需要被分割(地面点和无效点);
    //1） labelMat.at(i,j) = 0，初始值；
    //2） labelMat.at(i,j) = -1，无效点；
    //3）labelMat.at(thisIndX, thisIndY) = labelCount，平面点；
    //4）labelMat.at(allPushedIndX[i], allPushedIndY[i]) = 999999，需要舍弃的点，数量不到30。
    cv::Mat labelMat; // label matrix for segmentaiton marking

    //地面点的标志矩阵:1表示为地面点;0表示不是;-1表示无法判断
    //1） groundMat.at<int8_t>(i,j) = 0，初始值；
    //2） groundMat.at<int8_t>(i,j) = 1，有效的地面点；
    //3） groundMat.at<int8_t>(i,j) = -1，无效地面点；
    cv::Mat groundMat; // ground matrix for ground cloud marking

    //成功分割的簇的数量
    int labelCount;

    //一些临时使用的变量
    float startOrientation;
    float endOrientation;

    //自创的rosmsg来表示点云信息
    // segMsg点云信息(存储分割结果并用于发送)
    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    // segMsg{
    //     Header       header              //与接收到的点云数据header一致
    //     int32[]      startRingIndex      //segMsg点云中,每一行点云的起始和结束索引
    //     int32[]      endRingIndex
    //     float32      startOrientation    // 起始点与结束点的水平角度(atan(y,x))
    //     float32      endOrientation
    //     float32      orientationDiff     //以上两者的差
    //     bool[]       segmentedCloudGroundFlag //segMsg中点云的地面点标志序列(true:ground point)
    //     uint32[]     segmentedCloudColInd// segMsg中点云的cols序列
    //     float32[]    segmentedCloudRange //  segMsg中点云的range
    // }

    std_msgs::Header cloudHeader;

    // 四个pair集合(1,0)(-1,0)(0,1)(0,-1),用于在分割过程中检索点的深度图邻域
    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    // 分割过程中的临时变量

    // 用来记录一次聚类中取得的点的像素位置(x,y)
    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    //构造函数
    ImageProjection() :
            nh("~") {

        // 订阅主题 pointCloudTopic , 接收到点云信息后，调用函数cloudHandler.
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler,
                                                               this);

        // 多个发布者的声明
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info>("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);

        // 用于填充点云数据的无效点的赋值
        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        // 初始化各类参数以及分配内存
        allocateMemory();
        // 相关变量的初始化.
        resetParameters();
    }

    // 初始化各类参数以及分配内存
    void allocateMemory() {

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN * Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN * Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN * Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN * Horizon_SCAN, 0);

        // labelComponents函数中用到了这个矩阵
        // 该矩阵用于求某个点的上下左右4个邻接点
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = 1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = -1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN * Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN * Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN * Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN * Horizon_SCAN];
    }

    // 重置参数
    void resetParameters() {
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection() {}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
        // have "ring" channel in the cloud
        if (useCloudRing == true) {
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false) {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }
        }
    }

    //回调函数，对点云数据处理的主要部分
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // std_msgs::Header cloudHeader 获取消息的header
        // laserCloudIn(原始点云) 获得点云消息中去除无效点后的点云
        // useCloudRing 存储点云消息中的点云

        // 2. Start and end angle of a scan
        findStartEndAngle();
        // segMsg.startOrientation 记录起始点的角度(atan(y,x))
        // segMsg.endOrientation 记录结束点的角度(atan(y,x))
        // segMsg.orientationDiff 记录上面两者的差值

        // 3. Range image projection
        projectPointCloud();
        // a. 获取深度图 rangeMat
        // b. fullCloud 记录深度图点云
        // c. fullInfoCloud 记录深度图点云以及对应每个点的距离属性。

        // 4. Mark ground points
        groundRemoval();
        // a. 为地面点标志矩阵 groundMat 赋值: 1表示为地面点;0表示不是;-1表示无法判断
        // b. 为分割标志矩阵 labelMat 赋值: -1:不需要被分割的点(地面点和无效点);0:其他

        // 5. Point cloud segmentation
        cloudSegmentation();
        // a. 在深度图上对每个点进行分割,分割结果存储在 labelMat(分割标签矩阵) 中,其中确定值表示点所属类别,大值表示点未被分割,-1值表示地面点和无效点
        // b. 遍历每个点，   outlierCloud 记录经过降采样的未被分割点
        //     segmentedCloud 记录seg点云数据:包含被分割点和经过降采样的地面点
        //     segMsg.segmentedCloudGroundFlag 记录seg中点云的地面点标志序列
        //     segMsg.segmentedCloudColInd 记录seg中点云的cols
        //     segMsg.segmentedCloudRange 记录seg中点云的range
        // c. 为发布者 pubSegmentedCloudPure 准备数据 segmentedCloudPure
        //     segmentedCloudPure 存储被分割点点云，且每个点的i值为分割标志值。

        // 6. Publish all clouds
        publishCloud();
        // pubSegmentedCloudInfo 发布 segMsg
        // segMsg{
        //     Header header           //与接收到的点云数据header一致
        //     int32[] startRingIndex  //segMsg点云中,每一行点云的起始和结束索引
        //     int32[] endRingIndex
        //     float32 startOrientation// 起始点与结束点的水平角度(atan(y,x))
        //     float32 endOrientation
        //     float32 orientationDiff //以上两者的差
        //     bool[]    segmentedCloudGroundFlag //segMsg中点云的地面点标志序列(true:ground point)
        //     uint32[]  segmentedCloudColInd // segMsg中点云的cols序列
        //     float32[] segmentedCloudRange //  segMsg中点云的range
        // }
        // pubOutlierCloud 发布 msg{ //降采样的未被分割点云
        //     msg.header.stamp = cloudHeader.stamp;
        //     msg.header.frame_id = "base_link";
        //     outlierCloud;
        // }
        // pubSegmentedCloud 发布 msg{ //seg点云数据:包含被分割点和经过降采样的地面点
        //     msg.header.stamp = cloudHeader.stamp;
        //     msg.header.frame_id = "base_link";
        //     segmentedCloud;
        // }
        // pubFullCloud 发布 msg{ //一维形式存储深度图对应的点云
        //     msg.header.stamp = cloudHeader.stamp;
        //     msg.header.frame_id = "base_link";
        //     fullCloud;
        // }
        // pubGroundCloud 发布 msg{ //地面点集合
        //     msg.header.stamp = cloudHeader.stamp;
        //     msg.header.frame_id = "base_link";
        //     groundCloud;
        // }
        // pubSegmentedCloudPure 发布 msg{ // 存储被分割点点云，且每个点的i值为分割标志值
        //     msg.header.stamp = cloudHeader.stamp;
        //     msg.header.frame_id = "base_link";
        //     segmentedCloudPure;
        // }
        //  pubFullInfoCloud 发布 msg{ // 存储被分割点点云，且每个点的i值为分割标志值
        //     msg.header.stamp = cloudHeader.stamp;
        //     msg.header.frame_id = "base_link";
        //     fullInfoCloud;      //带距离值的深度图点云:与深度图点云存储一致的数据，但是其属性intensity记录的是距离值
        // }

        // 7. Reset parameters for next iteration
        resetParameters();

    }

    void findStartEndAngle() {
        // start and end orientation of this cloud
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                       laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    //逐一计算点云深度，并具有深度的点云保存至fullInfoCloud中
    void projectPointCloud() {
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize;
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i) {

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            //计算竖直方向上的点的角度以及在整个雷达点云中的哪一条水平线上
            // find the row and column index in the iamge for this point
            if (useCloudRing == true) {
                rowIdn = laserCloudInRing->points[i].ring;
            } else {
                verticalAngle =
                        atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            //出现异常角度则无视
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            //计算水平方向上点的角度与所在线数
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            //round是四舍五入
            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            //当前点与雷达的深度
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;
            //在rangeMat矩阵中保存该点的深度，保存单通道像素值
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            thisPoint.intensity = (float) rowIdn + (float) columnIdn / 10000.0;

            index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
        }
    }


    //利用不同的扫描圈来表示地面，进而检测地面是否水平。例如在源码中的七个扫描圈，每两个圈之间
    // 进行一次比较，角度相差10°以内的我们可以看做是平地。并且将扫描圈中的点加入到groundCloud点云
    void groundRemoval() {
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            for (size_t i = 0; i < groundScanInd; ++i) {

                lowerInd = j + (i) * Horizon_SCAN;
                upperInd = j + (i + 1) * Horizon_SCAN;

                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1) {
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i, j) = -1;
                    continue;
                }

                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10) {
                    groundMat.at<int8_t>(i, j) = 1;
                    groundMat.at<int8_t>(i + 1, j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        for (size_t i = 0; i < N_SCAN; ++i) {
            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (groundMat.at<int8_t>(i, j) == 1 || rangeMat.at<float>(i, j) == FLT_MAX) {
                    labelMat.at<int>(i, j) = -1;
                }
            }
        }
        if (pubGroundCloud.getNumSubscribers() != 0) {
            for (size_t i = 0; i <= groundScanInd; ++i) {
                for (size_t j = 0; j < Horizon_SCAN; ++j) {
                    if (groundMat.at<int8_t>(i, j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                }
            }
        }
    }

    //可以看到这是对点云分为地面点与可被匹配的四周被扫描的点，
    //将其筛选后分别纳入被分割点云
    void cloudSegmentation() {
        //这是在排除地面点与异常点之后，逐一检测邻点特征并生成局部特征
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i, j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {

            segMsg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                //如果是被认可的特征点或者是地面点，就可以纳入被分割点云
                if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1) {
                    // outliers that will not be used for optimization (always continue)
                    //离群点或异常点的处理
                    if (labelMat.at<int>(i, j) == 999999) {
                        if (i > groundScanInd && j % 5 == 0) {
                            outlierCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                            continue;
                        } else {
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i, j) == 1) {
                        //地面点云每隔5个点纳入被分割点云
                        if (j % 5 != 0 && j > 5 && j < Horizon_SCAN - 5)
                            continue;
                    }
                    //segMsg是自定义rosmsg
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i, j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i, j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;
        }

        // extract segmented cloud for visualization
        //如果在当前有节点订阅便将分割点云的几何信息也发布出去
        if (pubSegmentedCloudPure.getNumSubscribers() != 0) {
            for (size_t i = 0; i < N_SCAN; ++i) {
                for (size_t j = 0; j < Horizon_SCAN; ++j) {
                    if (labelMat.at<int>(i, j) > 0 && labelMat.at<int>(i, j) != 999999) {
                        segmentedCloudPure->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i, j);
                    }
                }
            }
        }
    }

    /*
    功能:以输入(row,col)为起点,进行聚类分割
    a. 以输入(row,col)为起点,获得聚类的所有点,存储在 allPushedIndSize(此次聚类的点的数量) (allPushedIndX,allPushedIndSize)(位置序列)
    b. 进行有效簇判断:单簇超过30个点，或者单簇超过5个点且跨越3个ring,视为有效簇
    c. 如果是有效簇: 为 labelMat 中的每个点赋予簇的标志值 labelCount,同时 labelCount++
    d. 如果不是有效簇：为 labelMat 中该簇的每个点赋予 999999,即不再用于以后的聚类
    */
    void labelComponents(int row, int col) {
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;

        //queueSize指的是在特征处理时还未处理好的点的数量，
        // 因此该while循环是在尝试检测该特定点的周围的点的几何特征
        while (queueSize > 0) {
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter) {
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));

                //该迭代器的first是0则是水平方向上的邻点，否则是竖直方向上的
                // alpha代表角度分辨率，
                // Y方向上角度分辨率是segmentAlphaY(rad)
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                // 通过下面的公式计算这两点之间是否有平面特征
                // atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
                //这个angle其实是该特定点与某邻点的连线与XOZ平面的夹角，这个夹角代表了局部特征的敏感性
                angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

                //如果夹角大于60°，则将这个邻点纳入到局部特征中，该邻点可以用来配准使用
                if (angle > segmentTheta) {

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        //当邻点数目达到30后，则该帧雷达点云的几何特征配置成功
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum) {
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;
        }
        // segment is valid, mark these points
        if (feasibleSegment == true) {
            ++labelCount;
        } else { // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i) {
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }


    //在我们计算的过程中参考系均为机器人自身参考系，frame_id自然是base_link。
    // 发布各类点云内容
    void publishCloud() {
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0) {
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0) {
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0) {
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0) {
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};


/*
	仅定义了ImageProjection IP,即调用构造函数 ImageProjection().
*/

int main(int argc, char **argv) {

    ros::init(argc, argv, "lego_loam");

    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
