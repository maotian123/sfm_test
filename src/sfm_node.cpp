#include <iostream>
#include <vector>


#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
cv::Mat prev_img, cur_img, forw_img;
std::vector<int> track_cnt;
std::vector<cv::Point2f> n_pts;
std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
std::map<int, cv::Point2f> cur_un_pts_map;
std::map<int, cv::Point2f> prev_un_pts_map;


extern bool PUB_THIS_FRAME;
const int MAX_CNT = 150;
const int MIN_DIST = 30;
std::vector<int> ids;

void readImage(const cv::Mat& img_);
void addPoints();




int main(int argv, char ** argc) {

    std::string img_dir_path = argc[1];
    const int img_num = atoi(argc[2]);
    
    std::vector<cv::Mat> sfm_img_;

    // for()

    for(int i = 0; i < img_num; ++i){
        const std::string img_path = img_dir_path + '/' + std::to_string(i) + ".jpeg";
        std::cout << "img_path : " << img_path << std::endl;
        cv::Mat img_ = cv::imread(img_path,0);
        // sfm_img_.push_back(img_);
        readImage(img_);
        // cv::imshow("img",img_);
        // cv::waitKey(0);
    }

    
    // std::cout << "img_dir_path : " << img_dir_path << std::endl;
    // std::cout << "img_num : " << img_num << std::endl;


    return 0;
}


void readImage(const cv::Mat& img_){
    cv::Mat img;
    // 均值化
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(img_, img);


    // cv::imshow("test",img);
    // cv::waitKey(0);

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if(cur_pts.size() > 0){
        std::vector<uchar> status;

        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        

    }

    for (auto &n : track_cnt)
        n++;

    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    //规定 150个 特征点,少于这个点就添加
    if(n_max_cnt > 0){
        cv::goodFeaturesToTrack(forw_img, n_pts,n_max_cnt,0.1,MIN_DIST);
    }
    else{
        n_pts.clear();
    }


    addPoints();


    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
}

void addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}


void undistortedPoints(){
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    /*
        focal length 5.6 mm 
        pixel w/h 2736
    */
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        liftProjective(a, b); //b为去畸变的世界坐标系归一化的点 
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        // updateid 函数会使得所有的特征点有单独的id号
        cur_un_pts_map.insert(std::make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
   
    prev_un_pts_map = cur_un_pts_map;
}


bool liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d & P){

    // static const cv::Mat K =  (cv::Mat_<double>( 3,3 ) << 701.8400268554688, 0.0, 486.79998779296875, 0.0, 702.0449829101562, 265.5050048828125, 0.0, 0.0, 1.0);
    // static const cv::Mat D =  (cv::Mat_<double>( 5,1 ) << -0.17445899546146393, 0.027615800499916077, 5.916510337211633e-11, 0.00015463300223927945, -2.4430599296465516e-05);
    static const double fx = 560;
    static const double fy = 560;
    static const double cx = 2736/2;
    static const double cy = 2736/2;
    

    // Lift points to normalised plane
    //X = (u- cx)/fx Y = (v - cy)/fy  points -> (col , row)
    const double u = p[0];
    const double v = p[1];
    const double X_distorted  = (p[0] - fx) / cx;
    const double Y_distorted  = (p[1] - fy) / cy;
    
    double X_undistorted, Y_undistorted;
    
    Eigen::Vector2d d_u;
    distortion(Eigen::Vector2d(X_distorted, Y_distorted),d_u);

    X_undistorted = X_distorted - d_u[0];
    Y_undistorted = Y_distorted - d_u[1];

    for(int i = 0; i < 8; ++i){  //循环去畸变
        distortion(Eigen::Vector2d(X_distorted, Y_distorted),d_u);

        X_undistorted = X_distorted - d_u[0];
        Y_undistorted = Y_distorted - d_u[1];
    }

    P << X_undistorted, Y_undistorted, 1;

    //  printf("X : %f, Y : %f\n",X ,Y );

}

bool distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d& d_u){
    static const double k1 = 0;
    static const double k2 = 0;
    static const double k3 = 0;
    static const double k4 = 0;


    double X_2, Y_2, XY, rho2, rad_dist;


    X_2 = p_u[0] * p_u[0];
    Y_2 = p_u[1] * p_u[1];
    XY = p_u[0] * p_u[1];
    rho2 = X_2 + Y_2;
    rad_dist = k1 * rho2 + k2*rho2 * rho2;
    d_u << p_u(0) * rad_dist + 2 * k3*XY + k4*(rho2 + 2 * X_2),
                    p_u(1) * rad_dist +  2 *k4*XY + k3*(rho2 + 2 * Y_2);
}
