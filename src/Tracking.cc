/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include "ORBmatcher.h"
#include "FramePublisher.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>
#include <fstream>
#include "geometry_msgs/Twist.h"

#define PI 3.14159265258979

//const char FROM[] = "Rulo/odom";
//const char TO[] = "Rulo/left_camera";

const char FROM[] = "myRobot/odom";
const char TO[] = "myRobot/left_camera";
const char CMD[] = "myRobot/cmd_vel";

using namespace std;


namespace ORB_SLAM
{

void Tracking::cmdCallback(const geometry_msgs::Twist& vel_cmd)
{
//ROS_INFO("I heard: [%s]", vel_cmd.linear.y);
    cout << "Twist Received " << endl;
    mV = vel_cmd.linear.x;
    mYawRate = vel_cmd.angular.z;

}

Tracking::Tracking(ORBVocabulary *pVoc, FramePublisher *pFramePublisher, MapPublisher *pMapPublisher, Map *pMap, string strSettingPath) : mState(NO_IMAGES_YET), mpORBVocabulary(pVoc), mpFramePublisher(pFramePublisher), mpMapPublisher(pMapPublisher), mpMap(pMap),
                                                                                                                                          mnLastRelocFrameId(0), mbPublisherStopped(false), mbReseting(false), mbForceRelocalisation(false), mbMotionModel(false)
{
    // Load camera parameters from settings file
    ofs.open("/home/liu/odom_with_odom_res.txt", std::ofstream::out);
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    DistCoef.copyTo(mDistCoef);

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = 18 * fps / 30;

    cout << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fastTh = fSettings["ORBextractor.fastTh"];
    int Score = fSettings["ORBextractor.nScoreType"];

    assert(Score == 1 || Score == 0);

    mpORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, Score, fastTh);

    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Fast Threshold: " << fastTh << endl;
    if (Score == 0)
        cout << "- Score: HARRIS" << endl;
    else
        cout << "- Score: FAST" << endl;

    // ORB extractor for initialization
    // Initialization uses only points from the finest scale level
    mpIniORBextractor = new ORBextractor(nFeatures * 2, 1.2, 8, Score, fastTh);

    int nMotion = fSettings["UseMotionModel"];
    mbMotionModel = nMotion;

    if (mbMotionModel)
    {
        mVelocity = cv::Mat::eye(4, 4, CV_32F);
        cout << endl
             << "Motion Model: Enabled" << endl
             << endl;
    }
    else
        cout << endl
             << "Motion Model: Disabled (not recommended, change settings UseMotionModel: 1)" << endl
             << endl;

    tf::Transform tfT;
    tfT.setIdentity();

    mTfBr.sendTransform(tf::StampedTransform(tfT, ros::Time::now(), "/ORB_SLAM/World", "/ORB_SLAM/Camera"));

    mV = 0;
    mYawRate = 0;
    try
    {
        mListener.waitForTransform(FROM, TO, ros::Time(0), ros::Duration(3.0));
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
    }
}



void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetKeyFrameDatabase(KeyFrameDatabase *pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

void Tracking::Run()
{
    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe("/stereo/left/image_raw", 1, &Tracking::GrabImage, this);
    ros::Subscriber odom_sub = nodeHandler.subscribe(CMD, 1, &Tracking::cmdCallback, this);
    ros::spin();
}

void computeAnglesFromMatrix(
    cv::Mat R,
    float &angle_x,
    float &angle_y,
    float &angle_z)
{

    float threshold = 0.001;

    if (abs(R.at<float>(2, 1) - 1.0) < threshold)
    { // R(2,1) = sin(x) = 1の時
        angle_x = PI / 2;
        angle_y = 0;
        angle_z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
    }
    else if (abs(R.at<float>(2, 1) + 1.0) < threshold)
    { // R(2,1) = sin(x) = -1の時
        angle_x = -PI / 2;
        angle_y = 0;
        angle_z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
    }
    else
    {
        angle_x = asin(R.at<float>(2, 1));
        angle_y = atan2(-R.at<float>(2, 0), R.at<float>(2, 2));
        angle_z = atan2(-R.at<float>(0, 1), R.at<float>(1, 1));
    }
}

cv::Mat computeMatrixFromAngles(
    float x,
    float y,
    float z)
{
    cv::Mat R = cv::Mat::zeros(3, 3, CV_32F);
    R.row(0).col(0) = cos(y) * cos(z) - sin(x) * sin(y) * sin(z);
    R.row(0).col(1) = -cos(x) * sin(z);
    R.row(0).col(2) = sin(y) * cos(z) + sin(x) * cos(y) * sin(z);
    R.row(1).col(0) = cos(y) * sin(z) + sin(x) * sin(y) * cos(z);
    R.row(1).col(1) = cos(x) * cos(z);
    R.row(1).col(2) = sin(y) * sin(z) - sin(x) * cos(y) * cos(z);
    R.row(2).col(0) = -cos(x) * sin(y);
    R.row(2).col(1) = sin(x);
    R.row(2).col(2) = cos(x) * cos(y);
    return R;
}

void Tracking::GrabImage(const sensor_msgs::ImageConstPtr &msg)
{

    cv::Mat im;

    mPre_stamp = mCur_stamp;
    mCur_stamp = msg->header.stamp.toSec();


    // Copy the ros image message to cv::Mat. Convert to grayscale if it is a color image.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    ROS_ASSERT(cv_ptr->image.channels() == 3 || cv_ptr->image.channels() == 1);

    if (cv_ptr->image.channels() == 3)
    {
        if (mbRGB)
            cvtColor(cv_ptr->image, im, CV_RGB2GRAY);
        else
            cvtColor(cv_ptr->image, im, CV_BGR2GRAY);
    }
    else if (cv_ptr->image.channels() == 1)
    {
        cv_ptr->image.copyTo(im);
    }

    if (mState == WORKING || mState == LOST)
        mCurrentFrame = Frame(im, cv_ptr->header.stamp.toSec(), mpORBextractor, mpORBVocabulary, mK, mDistCoef);
    else
        mCurrentFrame = Frame(im, cv_ptr->header.stamp.toSec(), mpIniORBextractor, mpORBVocabulary, mK, mDistCoef);

    if (mState != WORKING)
    {
        try
        {
            tf::StampedTransform transform;
            tf::Vector3 transpose;
            tf::Quaternion q;
            //"odomx", "base_footprintx"
            //
            mListener.lookupTransform(FROM, TO, ros::Time(0), transform);
            transpose = transform.getOrigin();
            q = transform.getRotation();

            // yaw (z-axis rotation)
            double ysqr = q.y() * q.y();
            double t3 = +2.0 * (q.w() * q.z() + q.x() * q.y());
            double t4 = +1.0 - 2.0 * (ysqr + q.z() * q.z());
            double yaw = std::atan2(t3, t4);
            mCurrentFrame.mYaw = yaw;
            mCurrentFrame.mTranspose = cv::Mat::zeros(3, 1, CV_32F);
            mCurrentFrame.mTranspose.at<float>(0) = -transpose.y();
            mCurrentFrame.mTranspose.at<float>(1) = transpose.x();
            mCurrentFrame.mTranspose.at<float>(2) = 0;
            mCurrentFrame.mYaw = yaw;

            //msg->header.stamp.toSec();
            //printf("tf_->x:%5.3f y:%5.3f z:%5.3f yaw:%5.3f\n", mCurrentFrame.mTranspose.at<float>(0), mCurrentFrame.mTranspose.at<float>(1), mCurrentFrame.mTranspose.at<float>(2), yaw);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
    }

    // Depending on the state of the Tracker we perform different tasks

    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    if (mState == NOT_INITIALIZED)
    {
        FirstInitialization();
    }
    else if (mState == INITIALIZING)
    {
        Initialize();
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial Camera Pose Estimation from Previous Frame (Motion Model or Coarse) or Relocalisation
        if (mState == WORKING && !RelocalisationRequested())
        {
            /*
            
            if (!mbMotionModel || mpMap->KeyFramesInMap() < 4 || mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                bOK = TrackPreviousFrame();
            }
            else
            {
                bOK = TrackWithMotionModel();
                if (!bOK)
                {
                    bOK = TrackPreviousFrame();
                }
            }*/

            bOK = TrackWithMotionModel();
            if (!bOK)
            {
                ROS_WARN("TrackPreviousFrame!");
                bOK = TrackPreviousFrame();
            }
        }
        else
        {
            bOK = Relocalisation();
        }

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (bOK)
            bOK = TrackLocalMap();

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            mpMapPublisher->SetCurrentCameraPose(mCurrentFrame.mTcw);

            if (NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (size_t i = 0; i < mCurrentFrame.mvbOutlier.size(); i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }

        if (bOK)
            mState = WORKING;
        else
            mState = LOST;

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                ROS_WARN("Reset!");
                Reset();
                return;
            }
        }

        // Update motion model
        if (mbMotionModel)
        {
                cv::Mat Rwc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * mCurrentFrame.mTcw.rowRange(0, 3).col(3);
                cout << "ORB:" << twc.at<float>(0) << "," << twc.at<float>(1) << endl;

            /*
            if (bOK && !mLastFrame.mTcw.empty())
            {
                cv::Mat LastRwc = mLastFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat Lasttwc = -LastRwc * mLastFrame.mTcw.rowRange(0, 3).col(3);
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                LastRwc.copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                Lasttwc.copyTo(LastTwc.rowRange(0, 3).col(3));

                mVelocity = mCurrentFrame.mTcw * LastTwc;

                float yaw_cl = mLastFrame.mYaw - mCurrentFrame.mYaw;

                if (yaw_cl < -PI)
                {
                    yaw_cl += 2 * PI;
                }
                else if (yaw_cl > PI)
                {
                    yaw_cl -= 2 * PI;
                }
                cv::Mat R_cl = computeMatrixFromAngles(0, 0, yaw_cl);
                cv::Mat R_cw = computeMatrixFromAngles(0, 0, -mCurrentFrame.mYaw);
                cv::Mat t_cl = R_cw * (mLastFrame.mTranspose - mCurrentFrame.mTranspose);

                mVelocity = cv::Mat::eye(4, 4, CV_32F);
                R_cl.copyTo(mVelocity.rowRange(0, 3).colRange(0, 3));
                t_cl.copyTo(mVelocity.rowRange(0, 3).col(3));

                cv::Mat tpre = mVelocity * mCurrentFrame.mTcw;
                cv::Mat tRwc = tpre.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat ttwc = -tRwc * tpre.rowRange(0, 3).col(3);


                cv::Mat Rwc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * mCurrentFrame.mTcw.rowRange(0, 3).col(3);

                //float secs = msg->header.stamp.toSec();
                //cout << secs << "," << twc.at<float>(0) << "," << twc.at<float>(1) << endl;
                //ofs << secs << "," << twc.at<float>(0) << "," << twc.at<float>(1) << endl;

                
                cout<<"============================================"<<endl;
                cout<<"mCurrentFrame.mTranspose"<<mCurrentFrame.mTranspose<<endl;
                cout<<"t_cl:"<<t_cl<<endl;
                cout<<"pre :"<<ttwc<<endl;
                cout<<"--------------------------------"<<endl;
                cout<<"mCurrentFrame.mYaw"<<mCurrentFrame.mYaw<<endl;
                cout<<"yaw_cl"<<yaw_cl<<endl;
                cout<<"mVelocity="<<mVelocity<<endl;
                cout<<"mCurrentFrame.mTcw="<<mCurrentFrame.mTcw<<endl;
                
                

            }
            else
                mVelocity = cv::Mat();*/
        }

        mLastFrame = Frame(mCurrentFrame);
    }

    // Update drawer
    mpFramePublisher->Update(this);

    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Rwc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * mCurrentFrame.mTcw.rowRange(0, 3).col(3);
        tf::Matrix3x3 M(Rwc.at<float>(0, 0), Rwc.at<float>(0, 1), Rwc.at<float>(0, 2),
                        Rwc.at<float>(1, 0), Rwc.at<float>(1, 1), Rwc.at<float>(1, 2),
                        Rwc.at<float>(2, 0), Rwc.at<float>(2, 1), Rwc.at<float>(2, 2));
        tf::Vector3 V(twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));

        tf::Transform tfTcw(M, V);

        mTfBr.sendTransform(tf::StampedTransform(tfTcw, ros::Time::now(), "ORB_SLAM/World", "ORB_SLAM/Camera"));

        
        float roll,pitch, yaw;

        computeAnglesFromMatrix(Rwc, roll, pitch, yaw);

        //printf("orb->x:%5.3f y:%5.3f z:%5.3f yaw:%5.3f\n",twc.at<float>(0), twc.at<float>(1), twc.at<float>(2), yaw);

        
        
    }
    //mpLocalMapper->Run();
}

void Tracking::FirstInitialization()
{
    //We ensure a minimum ORB features to continue, otherwise discard frame
    if (mCurrentFrame.mvKeys.size() > 100)
    {
        mInitialFrame = Frame(mCurrentFrame);
        mLastFrame = Frame(mCurrentFrame);
        mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
        for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
            mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

        if (mpInitializer)
            delete mpInitializer;

        mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

        mState = INITIALIZING;

        /*
        tf::StampedTransform transform;
        tf::Vector3 transpose;
        tf::Quaternion q;
        try
        {
            //"odomx", "base_footprintx"
            //
            mListener.lookupTransform(FROM, TO,
                                      ros::Time(0), transform);
            transpose = transform.getOrigin();
            q = transform.getRotation();

            // yaw (z-axis rotation)
            double ysqr = q.y() * q.y();
            double t3 = +2.0 * (q.w() * q.z() + q.x() * q.y());
            double t4 = +1.0 - 2.0 * (ysqr + q.z() * q.z());
            double yaw = std::atan2(t3, t4);

            mInitialFrame.mTranspose = cv::Mat::zeros(3, 1, CV_32F);
            mInitialFrame.mTranspose.at<float>(0) = -transpose.y();
            mInitialFrame.mTranspose.at<float>(1) = transpose.x();
            mInitialFrame.mTranspose.at<float>(2) = 0;
            mInitialFrame.mYaw = yaw;
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
        }
        */
    }
}

void Tracking::Initialize()
{
    // Check if current frame has enough keypoints, otherwise reset initialization process
    if (mCurrentFrame.mvKeys.size() <= 100)
    {
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        mState = NOT_INITIALIZED;
        return;
    }

    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < 100)
    {
        mState = NOT_INITIALIZED;
        return;
    }
    /*
    tf::StampedTransform transform;
    tf::Vector3 transpose;
    tf::Quaternion q;
    try
    {
        //"odomx", "base_footprintx"
        //
        mListener.lookupTransform(FROM, TO,
                                  ros::Time(0), transform);
        transpose = transform.getOrigin();
        q = transform.getRotation();

        // yaw (z-axis rotation)
        double ysqr = q.y() * q.y();
        double t3 = +2.0 * (q.w() * q.z() + q.x() * q.y());
        double t4 = +1.0 - 2.0 * (ysqr + q.z() * q.z());
        double yaw = std::atan2(t3, t4);
        mCurrentFrame.mTranspose = cv::Mat::zeros(3, 1, CV_32F);
        mCurrentFrame.mTranspose.at<float>(0) = -transpose.y();
        mCurrentFrame.mTranspose.at<float>(1) = transpose.x();
        mCurrentFrame.mTranspose.at<float>(2) = 0;
        mCurrentFrame.mYaw = yaw;
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
    }
    */

    float yaw_ci = mInitialFrame.mYaw - mCurrentFrame.mYaw;
    //cv::Mat t_ci = mCurrentFrame.mTranspose - mInitialFrame.mTranspose;

    if (yaw_ci < -PI)
    {
        yaw_ci += 2 * PI;
    }
    else if (yaw_ci > PI)
    {
        yaw_ci -= 2 * PI;
    }
    cv::Mat R_ci = computeMatrixFromAngles(0, 0, yaw_ci);
    cv::Mat R_cw = computeMatrixFromAngles(0, 0, -mCurrentFrame.mYaw);
    cv::Mat t_ci = -R_cw * (mCurrentFrame.mTranspose - mInitialFrame.mTranspose);

    cv::Mat Rcw2 = R_ci;
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = t_ci;

    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0, 3));
    tcw2.copyTo(Tcw2.col(3));

    cv::Mat Rcw1 = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat Rwc1 = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat tcw1 = cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat Tcw1 = cv::Mat::eye(3, 4, CV_32F);

    const float fx2 = mCurrentFrame.fx;
    const float fy2 = mCurrentFrame.fy;
    const float cx2 = mCurrentFrame.cx;
    const float cy2 = mCurrentFrame.cy;
    const float invfx2 = 1.0f / fx2;
    const float invfy2 = 1.0f / fy2;

    const float fx1 = mInitialFrame.fx;
    const float fy1 = mInitialFrame.fy;
    const float cx1 = mInitialFrame.cx;
    const float cy1 = mInitialFrame.cy;
    const float invfx1 = 1.0f / fx1;
    const float invfy1 = 1.0f / fy1;

    // Triangulate each match
    //mInitialFrame,mCurrentFrame
    //   SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    double base_line = cv::norm(tcw2);
    //printf("%f\n",base_line);
    //printf("tfi->x:%5.3f y:%5.3f z:%5.3f yaw:%5.3f\n",-transpose.y(), transpose.x(), transpose.z(), mCurrentFrame.mYaw);
    if (base_line > 0.1)
    {
        mvIniP3D.resize(mvIniMatches.size());
        for (size_t ikp = 0, iendkp = mvIniMatches.size(); ikp < iendkp; ikp++)
        {
            if (mvIniMatches[ikp] == -1)
                continue;

            const cv::KeyPoint &kp1 = mInitialFrame.mvKeysUn[ikp];
            const cv::KeyPoint &kp2 = mCurrentFrame.mvKeysUn[mvIniMatches[ikp]];

            //cv::line(mInitialFrame.im, kp1.pt, kp2.pt, cv::Scalar(0, 0, 255));
            //cv::line(mCurrentFrame.im, kp1.pt, kp2.pt, cv::Scalar(0, 0, 255));
            //cv::circle(mInitialFrame.im, kp1.pt, 2, cv::Scalar(0, 255, 0));
            //cv::circle(mCurrentFrame.im, kp2.pt, 2, cv::Scalar(0, 255, 0));

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            if (cosParallaxRays < 0 || cosParallaxRays > 0.9998)
                continue;

            // Linear Triangulation Method
            cv::Mat A(4, 4, CV_32F);
            A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
            A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
            A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
            A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

            cv::Mat w, u, vt;
            cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

            cv::Mat x3D = vt.row(3).t();

            if (x3D.at<float>(3) == 0)
                continue;

            // Euclidean coordinates
            x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0)
                continue;
            mvIniP3D[ikp] = cv::Point3f(x3Dt.at<float>(0),x3Dt.at<float>(1),x3Dt.at<float>(2));
            //printf("init point %5.3f %5.3f %5.3f\n",x3Dt.at<float>(0),x3Dt.at<float>(1),x3Dt.at<float>(2));
        }
        cv::imwrite("left.png",mInitialFrame.im);
        cv::imwrite("right.png",mCurrentFrame.im);
        CreateInitialMap(Rcw1, tcw1);
        cout<<"My Initial OK!"<<endl;
    }

    //printf("ini->x:%5.3f y:%5.3f z:%5.3f yaw:%5.3f\n",mInitialFrame.mTranspose.at<float>(0), mInitialFrame.mTranspose.at<float>(1), mInitialFrame.mTranspose.at<float>(2), mInitialFrame.mYaw);
    //printf("cur->x:%5.3f y:%5.3f z:%5.3f yaw:%5.3f\n",mCurrentFrame.mTranspose.at<float>(0), mCurrentFrame.mTranspose.at<float>(1), mCurrentFrame.mTranspose.at<float>(2), mCurrentFrame.mYaw);
    //printf("orb->x:%5.3f y:%5.3f z:%5.3f yaw:%5.3f\n",t_ci.at<float>(0), t_ci.at<float>(1), t_ci.at<float>(2), yaw_ci);

    /*cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
    

    if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
        {
            if(mvIniMatches[i]>=0 && !vbTriangulated[i])
            {
                mvIniMatches[i]=-1;
                nmatches--;
            }           
        }

        CreateInitialMap(Rcw,tcw);
    }*/
}

void Tracking::CreateInitialMap(cv::Mat &Rcw, cv::Mat &tcw)
{
    // Set Frame Poses
    mInitialFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    mCurrentFrame.mTcw = cv::Mat::eye(4, 4, CV_32F);
    Rcw.copyTo(mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3));
    tcw.copyTo(mCurrentFrame.mTcw.rowRange(0, 3).col(3));

    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    ROS_INFO("New Map created with %d points", mpMap->MapPointsInMap());

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    //float invMedianDepth = 1.0f / medianDepth;
    float invMedianDepth = 1.0f;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints() < 100)
    {
        ROS_INFO("Wrong initialization, reseting...");
        //Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);
    cout<<Tc2w<<endl;

    // Scale points
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.mTcw = pKFcur->GetPose().clone();
    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapPublisher->SetCurrentCameraPose(pKFcur->GetPose());

    mState = WORKING;
}

bool Tracking::TrackPreviousFrame()
{
    ORBmatcher matcher(0.9, true);
    vector<MapPoint *> vpMapPointMatches;

    // Search first points at coarse scale levels to get a rough initial estimate
    int minOctave = 0;
    int maxOctave = mCurrentFrame.mvScaleFactors.size() - 1;
    if (mpMap->KeyFramesInMap() > 5)
        minOctave = maxOctave / 2 + 1;

    int nmatches = matcher.WindowSearch(mLastFrame, mCurrentFrame, 200, vpMapPointMatches, minOctave);

    // If not enough matches, search again without scale constraint
    if (nmatches < 10)
    {
        nmatches = matcher.WindowSearch(mLastFrame, mCurrentFrame, 100, vpMapPointMatches, 0);
        if (nmatches < 10)
        {
            vpMapPointMatches = vector<MapPoint *>(mCurrentFrame.mvpMapPoints.size(), static_cast<MapPoint *>(NULL));
            nmatches = 0;
        }
    }

    mLastFrame.mTcw.copyTo(mCurrentFrame.mTcw);
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // If enough correspondeces, optimize pose and project points from previous frame to search more correspondences
    if (nmatches >= 10)
    {
        // Optimize pose with correspondences
        Optimizer::PoseOptimization(&mCurrentFrame);

        for (size_t i = 0; i < mCurrentFrame.mvbOutlier.size(); i++)
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
                mCurrentFrame.mvbOutlier[i] = false;
                nmatches--;
            }

        // Search by projection with the estimated pose
        nmatches += matcher.SearchByProjection(mLastFrame, mCurrentFrame, 15, vpMapPointMatches);
    }
    else //Last opportunity
        nmatches = matcher.SearchByProjection(mLastFrame, mCurrentFrame, 50, vpMapPointMatches);

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    if (nmatches < 10)
        return false;

    // Optimize pose again with all correspondences
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    for (size_t i = 0; i < mCurrentFrame.mvbOutlier.size(); i++)
        if (mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvpMapPoints[i] = NULL;
            mCurrentFrame.mvbOutlier[i] = false;
            nmatches--;
        }

    return nmatches >= 10;
}

bool Tracking::TrackWithMotionModel()
{
    static float x;
    static float y;
    ORBmatcher matcher(0.9, true);

    cv::Mat Rwc = mLastFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat twc = -Rwc * mLastFrame.mTcw.rowRange(0, 3).col(3);
    float roll, pitch, yaw;
    computeAnglesFromMatrix(Rwc, roll, pitch, yaw);

    float dx, dy;
    float dt = mCur_stamp - mPre_stamp;
    float r = sqrt(0.175*0.175 + 0.05*0.05);
    mVc = mV + mYawRate * r;
    if (mYawRate < 0.0001)
    {
        dx = mVc * dt * cos(yaw);
        dy = mVc * dt * sin(yaw);
    }
    else
    {
        dx = (mVc / mYawRate) * (sin(mYawRate * dt + yaw) - sin(yaw));
        dy = (mVc / mYawRate) * (-cos(mYawRate * dt + yaw) + cos(yaw));
    }
    //cout<<"dx:"<<dx<<endl;
    //cout<<"dy:"<<dy<<endl;
    x = x + dx;
    y = y + dy;

    float tmp;
    tmp = dx;
    dx = -dy;
    dy = tmp;

    //cout<<"----------"<<endl;
    cout << "odm:" << -y<< "," << x << endl;
    // Compute current pose by motion model
    
    mCurrentFrame.mTcw = mLastFrame.mTcw;

/*
    cout<<"------------------------"<<endl;
    cout<<"vec:"<<mVelocity<<endl;
    cout<<"Last mTcw"<<mLastFrame.mTcw<<endl;
    cout<<"pre mTcw"<<mCurrentFrame.mTcw<<endl;
    */
    


    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 15);
    if (nmatches < 20)
        return false;

    // Optimize pose with all correspondences
    Optimizer::PoseOptimization(&mCurrentFrame);
    //cout<<"bef mTcw"<<mCurrentFrame.mTcw<<endl;
    // Discard outliers
    for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
                mCurrentFrame.mvbOutlier[i] = false;
                nmatches--;
            }
        }
    }

    return nmatches >= 10;
}

bool Tracking::TrackLocalMap()
{
    // Tracking from previous frame or relocalisation was succesfull and we have an estimation
    // of the camera pose and some map points tracked in the frame.
    // Update Local Map and Track

    // Update Local Map
    UpdateReference();

    // Search Local MapPoints
    SearchReferencePointsInFrustum();

    // Optimize Pose
    mnMatchesInliers = Optimizer::PoseOptimization(&mCurrentFrame);

    // Update MapPoints Statistics
    for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++)
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (!mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 20)//mod_by_liu 50 -> 20
        return false;

    if (mnMatchesInliers < 20)//mod_by_liu 30 -> 20
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    // Not insert keyframes if not enough frames from last relocalisation have passed
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mpMap->KeyFramesInMap() > mMaxFrames)
        return false;

    // Reference KeyFrame MapPoints
    int nRefMatches = mpReferenceKF->TrackedMapPoints();

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    //const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    //const bool c1b = mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle;
    // Condition 2: Less than 90% of points than reference keyframe and enough inliers
    const bool c2 = mnMatchesInliers < nRefMatches * 0.6 && mnMatchesInliers > 15; // mod_by_liu 0.9 -> 0.6

    //if ((c1a || c1b) && c2)// mod_by_liu 0
    if ( c2)
    {
        // If the mapping accepts keyframes insert, otherwise send a signal to interrupt BA, but not insert yet
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    mpLocalMapper->InsertKeyFrame(pKF);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchReferencePointsInFrustum()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = NULL;
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mbTrackInView = false;
            }
        }
    }

    mCurrentFrame.UpdatePoseMatrices();

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateReference()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateReferenceKeyFrames();
    UpdateReferencePoints();
}

void Tracking::UpdateReferencePoints()
{
    mvpLocalMapPoints.clear();

    for (vector<KeyFrame *>::iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateReferenceKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (size_t i = 0, iend = mCurrentFrame.mvpMapPoints.size(); i < iend; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    int max = 0;
    KeyFrame *pKFmax = NULL;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }
    }

    mpReferenceKF = pKFmax;
}

bool Tracking::Relocalisation()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalisation is performed when tracking is lost and forced at some stages during loop closing
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs;
    if (!RelocalisationRequested())
        vpCandidateKFs = mpKeyFrameDB->DetectRelocalisationCandidates(&mCurrentFrame);
    else // Forced Relocalisation: Relocate against local window around last keyframe
    {
        boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
        mbForceRelocalisation = false;
        vpCandidateKFs.reserve(10);
        vpCandidateKFs = mpLastKeyFrame->GetBestCovisibilityKeyFrames(9);
        vpCandidateKFs.push_back(mpLastKeyFrame);
    }

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (size_t i = 0; i < vpCandidateKFs.size(); i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (size_t i = 0; i < vpCandidateKFs.size(); i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                for (size_t j = 0; j < vbInliers.size(); j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (size_t io = 0, ioend = mCurrentFrame.mvbOutlier.size(); io < ioend; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = NULL;

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (size_t ip = 0, ipend = mCurrentFrame.mvpMapPoints.size(); ip < ipend; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (size_t io = 0; io < mCurrentFrame.mvbOutlier.size(); io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::ForceRelocalisation()
{
    boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
    mbForceRelocalisation = true;
    mnLastRelocFrameId = mCurrentFrame.mnId;
}

bool Tracking::RelocalisationRequested()
{
    boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
    return mbForceRelocalisation;
}

void Tracking::Reset()
{
    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbPublisherStopped = false;
        mbReseting = true;
    }

    // Wait until publishers are stopped
    ros::Rate r(500);
    /*while(1)
    {
        {
            boost::mutex::scoped_lock lock(mMutexReset);
            if(mbPublisherStopped)
                break;
        }
        r.sleep();
    }*/

    // Reset Local Mapping
    mpLocalMapper->RequestReset();
    // Reset Loop Closing
    mpLoopClosing->RequestReset();
    // Clear BoW Database
    mpKeyFrameDB->clear();
    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NOT_INITIALIZED;

    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbReseting = false;
    }
}

void Tracking::CheckResetByPublishers()
{
    bool bReseting = false;

    {
        boost::mutex::scoped_lock lock(mMutexReset);
        bReseting = mbReseting;
    }

    if (bReseting)
    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbPublisherStopped = true;
    }

    // Hold until reset is finished
    ros::Rate r(500);
    while (1)
    {
        {
            boost::mutex::scoped_lock lock(mMutexReset);
            if (!mbReseting)
            {
                mbPublisherStopped = false;
                break;
            }
        }
        r.sleep();
    }
}

} //namespace ORB_SLAM
