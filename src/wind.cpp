#include <iostream>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <numeric>
#include <chrono>
using namespace std;
using namespace cv;
using namespace cv::ml;


//Ptr<SVM> svm=SVM::create();

int cnnt=0;
//#define FLOODFILL
Point waixin(Point a,Point b,Point c)
{
double x0,y0;
x0=((pow(a.x,2)-pow(b.x,2)+pow(a.y,2)-pow(b.y,2))*(a.y-c.y)-(pow(a.x,2)-pow(c.x,2)+pow(a.y,2)-pow(c.y,2))*(a.y-b.y))/(2*(a.y-c.y)*(a.x-b.x)-2*(a.y-b.y)*(a.x-c.x));
y0=((pow(a.x,2)-pow(b.x,2)+pow(a.y,2)-pow(b.y,2))*(a.x-c.x)-(pow(a.x,2)-pow(c.x,2)+pow(a.y,2)-pow(c.y,2))*(a.x-b.x))/(2*(a.y-b.y)*(a.x-c.x)-2*(a.y-c.y)*(a.x-b.x));
return Point(x0,y0);
   
}
static bool CircleInfo2(std::vector<cv::Point2f>& pts, cv::Point2f& center, float& radius)
{
    center = cv::Point2d(0, 0);
    radius = 0.0;
    if (pts.size() < 3) return false;;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumX3 = 0.0;
    double sumY3 = 0.0;
    double sumXY = 0.0;
    double sumX1Y2 = 0.0;
    double sumX2Y1 = 0.0;
    const double N = (double)pts.size();
    for (int i = 0; i < pts.size(); ++i)
    {
        double x = pts.at(i).x;
        double y = pts.at(i).y;
        double x2 = x * x;
        double y2 = y * y;
        double x3 = x2 *x;
        double y3 = y2 *y;
        double xy = x * y;
        double x1y2 = x * y2;
        double x2y1 = x2 * y;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumY2 += y2;
        sumX3 += x3;
        sumY3 += y3;
        sumXY += xy;
        sumX1Y2 += x1y2;
        sumX2Y1 += x2y1;
    }
    double C = N * sumX2 - sumX * sumX;
    double D = N * sumXY - sumX * sumY;
    double E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX;
    double G = N * sumY2 - sumY * sumY;
    double H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY;

    double denominator = C * G - D * D;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double a = (H * D - E * G) / (denominator);
    denominator = D * D - G * C;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double b = (H * C - E * D) / (denominator);
    double c = -(a * sumX + b * sumY + sumX2 + sumY2) / N;

    center.x = a / (-2);
    center.y = b / (-2);
    radius = std::sqrt(a * a + b * b - 4 * c) / 2;
    return true;
}


double getDistance(Point A,Point B)
{
    double dis;
    dis=pow((A.x-B.x),2)+pow((A.y-B.y),2);
    return sqrt(dis);
}
string int_to_str(int i)
{
    stringstream s;
    s<<i;
    return s.str();
}
//模板匹配
double TemplateMatch(cv::Mat image, cv::Mat tepl, cv::Point &point, int method)
{
    int result_cols =  image.cols -tepl.cols +1 ;
    int result_rows = image.rows -tepl.rows +1;
//    cout <<result_cols<<" "<<result_rows<<endl;
    cv::Mat result = cv::Mat( result_cols, result_rows, CV_32FC1 );
    cv::matchTemplate( image, tepl, result, method );

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    switch(method)
    {
    case CV_TM_SQDIFF:
    case CV_TM_SQDIFF_NORMED:
        point = minLoc;
        return minVal;

    default:
        point = maxLoc;
        return maxVal;

    }
}
Mat templ[9];
Mat drawcircle;

//Point2f cc2=Point2f(60,60);
int main()
{
  //  svm=SVM::load("../model/SVM.xml");
    VideoCapture cap("../video/wind_mill2.avi");
    Mat image,binary;
    int stateNum = 4;
    int measureNum = 2;
    KalmanFilter KF(stateNum, measureNum, 0);//这里没有设置控制矩阵B，默认为零
    //Mat processNoise(stateNum, 1, CV_32F);
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
    KF.transitionMatrix = (Mat_<float>(stateNum, stateNum) << 1, 0, 1, 0,//A 状态转移矩阵
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    
    Point2f cc=Point2f(0,0);
 
    setIdentity(KF.measurementMatrix);//H=[1,0,0,0;0,1,0,0] 测量矩阵
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q高斯白噪声，单位阵
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R高斯白噪声，单位阵
    setIdentity(KF.errorCovPost, Scalar::all(1));//P后验误差估计协方差矩阵，初始化为单位阵
    randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));//初始化状态为随机值
    
     for(int i=1;i<=8;i++)
    {
        templ[i]=imread("../template/template"+int_to_str(i)+".jpg",IMREAD_GRAYSCALE);
    }
    drawcircle=imread("1.jpg");
    for(;;){
        cap.read(image);
        vector<Point2f>circle_center;
       
       // Mat drawcircle=Mat(image.rows*0.5,image.cols*0.5, CV_8UC3, Scalar(0, 0, 0));
       // imwrite("1.jpg",drawcircle);

        image.copyTo(binary);
        resize(image,image,Size(binary.cols*0.5,binary.rows*0.5));
        resize(binary,binary,Size(binary.cols*0.5,binary.rows*0.5));
        vector<Mat> imgchannels;
        split(image,imgchannels);
        Mat midimage=imgchannels.at(2)-imgchannels.at(0);
        //cvtColor(image,image,COLOR_BGR2GRAY);
        threshold(midimage, midimage, 140, 255, THRESH_BINARY);      //80,wind_mill1.avi
        int structElementSize=1;
        Mat element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
        dilate(midimage,midimage,element);

      /*  
        structElementSize=3;
        element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
        morphologyEx(midimage,midimage, MORPH_CLOSE, element);
        imshow("kaicaozuo",midimage);
    */  vector<Mat>saver;
        vector<vector<Point> >contours2;
        vector<Vec4i> hierarchy2;
        imshow("erhzihua",midimage);





        findContours(midimage,contours2,hierarchy2,CV_RETR_TREE,CHAIN_APPROX_SIMPLE);
        if(contours2.size()>0)
        {

            for (size_t i = 0; i < contours2.size(); i++)
            {
                double area = contourArea(contours2[i]);
                if (area < 350 || 1e4 < area) continue;
                
                RotatedRect rect=minAreaRect(contours2[i]);
                /*
                Point2f center = rect.center;//外接矩形中心点坐标
	            Mat rot_mat = getRotationMatrix2D(center, rect.angle, 1.0);//求旋转矩阵
	            Mat rot_image;
	            Size dst_sz(midimage.size());
	            warpAffine(midimage, rot_image, rot_mat, dst_sz);//原图像旋转
	            imshow("rot_image", rot_image);
	            Mat roi (rot_image,Rect(center.x-(rect.size.width/2), center.y-(rect.size.height/2), rect.size.width, rect.size.height));//提取ROI
	            imshow("result", roi);
*/
        //    /     Mat roi(midimage,rect);
               // float aim = rect.size.height/rect.size.width;
                Point2f vertices[4];      //定义4个点的数组
                rect.points(vertices);   //将四个点存储到vertices数组中
                for (int j = 0; j < 4; j++)
                line(binary, vertices[j], vertices[(j+1)%4], Scalar(255,0,0),3);
                
               // imshow("chuli",binary);
                //saver.push_back(vertices);
                //为透视变换做准备
                Point2f srcRect[3];
                Point2f dstRect[3];

                double width;
                double height;

                //矫正提取的叶片的宽高
                width=getDistance(vertices[0],vertices[1]);
                height=getDistance(vertices[1],vertices[2]);
                if(width>height)
                {
                    srcRect[0]=vertices[0];
                    srcRect[1]=vertices[1];
                    srcRect[2]=vertices[2];
                }   
                else
                {
                    swap(width,height);
                    srcRect[0]=vertices[1];
                    srcRect[1]=vertices[2];
                    srcRect[2]=vertices[3];
                }

                //double area=height*width;
                //if(area>5000){

                dstRect[0]=Point2f(0,0);
                dstRect[1]=Point2f(width,0);
                dstRect[2]=Point2f(width,height);
                //透视变换，矫正成规则矩形
                Mat warp_mat=getAffineTransform(srcRect,dstRect);
                Mat warp_dst_map;
                warpAffine(midimage,warp_dst_map,warp_mat,warp_dst_map.size());
                // 提取扇叶图片
                Mat testim;
                testim = warp_dst_map(Rect(0,0,width,height));
               
                /*
                    resize(testim,testim,Size(42,20));
                    string s="leaf"+int_to_str(cnnt)+".jpg";
                    cnnt++;
                    imwrite("../img/"+s,testim);
                */
                cv::Point matchLoc;
                double value;
                Mat tmp1;
                    resize(testim,tmp1,Size(42,20));
                    vector<double> Vvalue1;
                    vector<double> Vvalue2;
                    for(int j=1;j<=6;j++)
                    {
                        value = TemplateMatch(tmp1, templ[j], matchLoc, CV_TM_CCOEFF_NORMED);
                        Vvalue1.push_back(value);
                    }
                    for(int j=7;j<=8;j++)
                    {
                        value = TemplateMatch(tmp1, templ[j], matchLoc, CV_TM_CCOEFF_NORMED);
                        Vvalue2.push_back(value);
                    }
                    int maxv1=0,maxv2=0;

                    for(int t1=0;t1<6;t1++)
                    {
                        if(Vvalue1[t1]>Vvalue1[maxv1])
                        {
                            maxv1=t1;
                        }
                    }
                    for(int t2=0;t2<2;t2++)
                    {
                        if(Vvalue2[t2]>Vvalue2[maxv2])
                        {
                            maxv2=t2;
                        }
                    }  
                
                


                if(Vvalue1[maxv1]>Vvalue2[maxv2]&&Vvalue1[maxv1]>0.6)
                {
                  //  cout<<"###########test success##############"<<endl;
                     if(hierarchy2[i][2]>=0)
                     {
                    RotatedRect rect_tmp=minAreaRect(contours2[hierarchy2[i][2]]);
                    
                  //  if (rect_tmp.size.height<rect_tmp.size.width)
                 //   {
                  //      swap(rect_tmp.size.height,rect_tmp.size.width);
                  //  }
                  //  float aim = rect_tmp.size.height/rect_tmp.size.width;
                  //  if(aim > 1.7 && aim < 2.6)
                  //  {
                    
                    Point2f Pnt[4];
                    rect_tmp.points(Pnt);

                    /*
                    const float maxHWRatio=0.7153846;
                    const float maxArea=500;
                    const float minArea=190;
                
                    float width=rect_tmp.size.width;
                    float height=rect_tmp.size.height;
                    if(height>width)
                    swap(height,width);
                    float area=width*height;
                    if(height/width>maxHWRatio||area>maxArea||area<minArea)
                    continue;
              
                 */
                    circle(binary,rect_tmp.center,1,Scalar(255,0,0),5);
                    putText(binary,"arrow",Point(rect_tmp.center.x-10,rect_tmp.center.y-10),FONT_HERSHEY_SIMPLEX,1,Scalar(10,255,30),3,8);
                   // circle(drawcircle,rect_tmp.center,1,Scalar(255,255,255),5);
                  /*
                    if(circle_center.size()<30)
                    {
                       circle_center.push_back(rect_tmp.center);
                    }
                    
                    else
                    {
                        float R;
                        //得到拟合的圆心
                       CircleInfo2(circle_center,cc,R);
                        
                       circle(binary,cc,1,Scalar(255,0,0),2);
                     //circle_center.erase(circle_center.begin());
                       // cout<<endl<<"center "<<cc.x<<" , "<<cc.y<<endl;
                    }
                    
                  //  }
                   
               */
                    }
                }
            
              
            }

            for (size_t i = 0; i < contours2.size(); i++)
                {
                if(hierarchy2[i][2]>=0)
                {
                    RotatedRect rect2=minAreaRect(contours2[hierarchy2[i][2]]);
                    

                    Point2f P[4];
                    rect2.points(P);
                    const float maxHWRatio=0.7153846;
                    const float maxArea=500;
                    const float minArea=200;
                
                    float width=rect2.size.width;
                    float height=rect2.size.height;
                    if(height>width)
                    swap(height,width);
                    float area=width*height;
                    if(height/width>maxHWRatio||area>maxArea||area<minArea)
                    continue;
        
   
                   // circle(binary,rect2.center,1,Scalar(255,0,0),5);
                  //  putText(binary,"arrow",Point(rect_tmp.center.x-10,rect_tmp.center.y-10),FONT_HERSHEY_SIMPLEX,1,Scalar(10,255,30),3,8);
                   // circle(drawcircle,rect_tmp.center,1,Scalar(255,255,255),5);
                  
                    if(circle_center.size()<3)
                    {
                       circle_center.push_back(rect2.center);
                    }
                    
                    else
                    {
                        float R;
                        //得到拟合的圆心
                      // CircleInfo2(circle_center,cc,R);
                       cc =waixin(circle_center[0],circle_center[1],circle_center[2]);
                       cc =Point(cc.x-10,cc.y-32);//矫正
                       circle(binary,cc,1,Scalar(0,255,0),10);
                     //circle_center.erase(circle_center.begin());
                       // cout<<endl<<"center "<<cc.x<<" , "<<cc.y<<endl;
                    }
                    
                  //  }
                   
               
                }
                }
            // imshow("dajizhongxin",binary);
            
        }

#ifdef FLOODFILL
        floodFill(midimage,Point(5,50),Scalar(255),0,FLOODFILL_FIXED_RANGE);
       // imshow("image3",image);
        threshold(midimage, midimage, 80, 255, THRESH_BINARY_INV);
       // imshow("image4",image);
        vector<vector<Point> > contours;
        findContours(midimage, contours, RETR_LIST, CHAIN_APPROX_NONE);
        for (size_t i = 0; i < contours.size(); i++)
        {

            vector<Point> points;
            double area = contourArea(contours[i]);
            if (area < 50 || 1e4 < area) continue;
            drawContours(midimage, contours, static_cast<int>(i), Scalar(0), 2);

            points = contours[i];
            RotatedRect rrect = fitEllipse(points);
            cv::Point2f* vertices = new cv::Point2f[4];
            rrect.points(vertices);

            float aim = rrect.size.height/rrect.size.width;
            if(aim > 1.7 && aim < 2.6)
            {
                /*
                Point center;
                for (int j = 0; j < 4; j++)
                {
                    //line(binary, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0),4);
                    center.x+=vertices[j].x;
                    center.y+=vertices[j].y;

                }
                center.x=center.x/4;
                center.y=center.y/4;
            */
                circle(binary, rrect.center, 3, Scalar(255, 0, 0), -1);
            }
        
        }  
#endif
         /*
               float middle = 100000;

                for(size_t j = 1;j < contours.size();j++)
                {

                    vector<Point> pointsA;
                    double area = contourArea(contours[j]);
                    if (area < 50 || 1e4 < area) continue;

                    pointsA = contours[j];

                    RotatedRect rrectA = fitEllipse(pointsA);

                    float aimA = rrectA.size.height/rrectA.size.width;

                    if(aimA > 3.0)
                    {
                    float distance = sqrt((rrect.center.x-rrectA.center.x)*(rrect.center.x-rrectA.center.x)+
                                          (rrect.center.y-rrectA.center.y)*(rrect.center.y-rrectA.center.y));

                    if (middle > distance)
                        middle = distance;
                    }
                }
                if( middle > 60)
                {                               //这个距离也要根据实际情况调,和图像尺寸和物体远近有关。
                    cv::circle(binary,Point(rrect.center.x,rrect.center.y),15,cv::Scalar(0,0,255),4);
                    Mat prediction = KF.predict();
                    Point predict_pt = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));

                    measurement.at<float>(0) = (float)rrect.center.x;
                    measurement.at<float>(1) = (float)rrect.center.y;
                    KF.correct(measurement);

                    circle(binary, predict_pt, 3, Scalar(34, 255, 255), -1);

                    rrect.center.x = (int)prediction.at<float>(0);
                    rrect.center.y = (int)prediction.at<float>(1);

                }
            }
        */
            
           
       
        imshow("frame",binary);
        imshow ("aaa",drawcircle);
     //   imshow("Original", midimage);
        char c=waitKey(30);
        if(c==27)
        {
            break;
        }
    }
}


