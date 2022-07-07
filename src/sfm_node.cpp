#include <iostream>
#include <vector>
#include <memory>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/eigen.hpp>


#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

using namespace Eigen;
using namespace std;

const int MAX_CNT = 150;
const int MIN_DIST = 30;
int COL;
int ROW;
bool PUB_THIS_FRAME = true;
int frame_count = 0;
bool init_pub = 0;

int WINDOW_SIZE = 10;
std::list<FeaturePerId> feature;

struct IMG_MSG {
    double header;
    std::vector<Eigen::Vector3d> points;
    std::vector<int> id_of_point;
    std::vector<float> u_of_point;
    std::vector<float> v_of_point;
    std::vector<float> velocity_x_of_point;
    std::vector<float> velocity_y_of_point;
};

typedef std::shared_ptr <IMG_MSG const > ImgConstPtr;
std::queue<ImgConstPtr> feature_buf;

struct SFMFeature
{
    bool state;
    int id;
    std::vector<std::pair<int,Eigen::Vector2d>> observation;
    double position[3];
    double depth;
};



struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
    public:
        GlobalSFM();
        bool construct(int frame_num, Eigen::Quaterniond* q, Eigen::Vector3d* T, int l,
                const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
                std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points);

    private:
        bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<SFMFeature> &sfm_f);

        void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
        void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
                                int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                std::vector<SFMFeature> &sfm_f);

        int feature_num;
};

GlobalSFM::GlobalSFM(){}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	// std::cout << " first matrix : " << R_initial << std::endl;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	// std::cout << " after pnp : " << R_initial << std::endl;
	P_initial = T_pnp;
	return true;

}

//frame0 第i帧图像 Pose0 传进来一般是单位帧 frame1 最后一帧图像 pose1 第一帧最后一帧的位姿
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	//feature_num 特征点的总共数量
    //
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		//每个feature所在图像帧的归一化坐标
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			//判断start frame是否fram0
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			//三角化 求解路标点 在相机坐标下的 参考 https://blog.csdn.net/hltt3838/article/details/105331457
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

/**
 * 对窗口中每个图像帧求解sfm问题，得到所有图像帧相对于参考帧的旋转四元数Q、平移向量T和特征点坐标sfm_tracked_points
 * 参数frame_num的值为frame_count + 1
 * 传入的参数l就是参考帧的index
 * */

bool GlobalSFM::construct(int frame_num, Eigen::Quaterniond* q, Eigen::Vector3d* T, int l,
			  const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
			  std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	
	//q为四元数数组，大小为frame_count+1
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	//T为平移量数组，大小为frame_count+1
	T[l].setZero();
	// std::cout << " frame_num : " << frame_num << std::endl;
    q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;

	//rotate to cam frame
	Eigen::Matrix3d c_Rotation[frame_num];
	Eigen::Vector3d c_Translation[frame_num];
	Eigen::Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	// std::cout << "relative_R : " << relative_R << std::endl <<
	// 						"relative_T : " << relative_T << std::endl;
	// std::cout << " Pose[l ] : " <<  Pose[l] << std::endl << " Pose[frame_num - 1] : " << Pose[frame_num - 1] << std::endl;
	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Eigen::Matrix3d R_initial = c_Rotation[i - 1];
			Eigen::Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result 
		// std::cout << " Pose[i] : " <<  Pose[i] << std::endl << " Pose[frame_num - 1] : " << Pose[frame_num - 1] << std::endl;
		// std::cout << " frame_num - 1 : " << frame_num - 1 << std::endl;
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Eigen::Matrix3d R_initial = c_Rotation[i + 1];
		Eigen::Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Eigen::Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Eigen::Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}


class FeaturePerFrame
{
    public:
        FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point)
        {
            point.x() = _point(0);
            point.y() = _point(1);
            point.z() = _point(2);
            uv.x() = _point(3);
            uv.y() = _point(4);
            velocity.x() = _point(5);
            velocity.y() = _point(6);
        }
        Eigen::Vector3d point;
        Eigen::Vector2d uv;
        Eigen::Vector2d velocity;
        double z;
        bool is_used;
        double parallax;
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        double dep_gradient;
};

class FeaturePerId
{
    public:
        const int feature_id;
        int start_frame;
        std::vector<FeaturePerFrame> feature_per_frame;

        int used_num;
        bool is_outlier;
        bool is_margin;
        double estimated_depth;
        int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

        Eigen::Vector3d gt_p;

        FeaturePerId(int _feature_id, int _start_frame)
            : feature_id(_feature_id), start_frame(_start_frame),
                used_num(0), estimated_depth(-1.0), solve_flag(0)
        {
        }

        int endFrame();
};

class FeatureTraker{
    public:
        FeatureTraker() = default;

        void readImage(const cv::Mat& img_);
        
        bool updateID(unsigned int i);
        

    private:
        
        bool inBorder(const cv::Point2f &pt);
        void addPoints();
        void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
        void reduceVector(std::vector<int> &v, std::vector<uchar> status);
        void setMask();
        void undistortedPoints();
        void rejectWithF();

        //归一化 去畸变
        bool liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d & P);

        bool distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d& d_u);
        

    public:
        std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
        cv::Mat prev_img, cur_img, forw_img;
        std::vector<int> track_cnt;
        std::vector<cv::Point2f> n_pts;
        std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
        std::map<int, cv::Point2f> cur_un_pts_map;
        std::map<int, cv::Point2f> prev_un_pts_map;
        cv::Mat mask;
        std::vector<int> ids;
        int n_id = 0;

    private:
    
        int feature_num_id = 0;
};

void FeatureTraker::readImage(const cv::Mat& img_){
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
        
        for (int i = 0; i < int(forw_pts.size()); i++)
                    if (status[i] && !inBorder(forw_pts[i]))
                        status[i] = 0;

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);

    }
    //本质矩阵 去除不好的关键点
    rejectWithF();

    setMask();
    
    for (auto &n : track_cnt)
        n++;

    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    //规定 150个 特征点,少于这个点就添加
    if(n_max_cnt > 0){
        cv::goodFeaturesToTrack(forw_img, n_pts,n_max_cnt,0.1,MIN_DIST,mask);
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

void FeatureTraker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

bool FeatureTraker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTraker::undistortedPoints(){
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


bool FeatureTraker::liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d & P){

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

bool FeatureTraker::distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d& d_u){
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

bool FeatureTraker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void FeatureTraker::reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];


    v.resize(j);
}

void FeatureTraker::reduceVector(std::vector<int> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTraker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");
        
        std::vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = 560 * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = 560 * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = 560 * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = 560 * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        std::vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, 1.0, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        
    }
}

void FeatureTraker::setMask()
{
    
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    // prefer to keep features that are tracked for long time
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(std::make_pair(track_cnt[i], std::make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            // cv::namedWindow("window", cv::WINDOW_FREERATIO);
            // cv::imshow("window",mask);
            // cv::waitKey(1);
        }
    }
}




//用于计算视差
double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Eigen::Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Eigen::Vector3d p_i = frame_i.point;
    Eigen::Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    
    double dep_i = p_i(2);

    // std::cout << " dep_i : " << dep_i << std::endl;
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;
    // std::cout << "1: " <<du_comp * du_comp + dv_comp * dv_comp << "  2 :" << "du * du + dv * dv"  << du * du + dv * dv << std::endl;
    ans = std::max(ans, sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}


std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r)
{
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(std::make_pair(a, b));
        }
    }
    return corres;
}
/*
这里计算出滑动窗口中第一个满足条件的帧和最新帧之间的旋转和平移之后，还要判断内点数是否大于12，大于12才认为计算出的旋转和平移量是有效的。
*/
bool solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres, Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation)
{
    if (corres.size() >= 15)
    {
        std::vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}

/**
 * 这里的第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，
 * 会作为参考帧到下面的全局sfm使用，得到的Rt为当前帧到第l帧的坐标系变换Rt
 * 该函数判断滑动窗口中第一个到窗口最后一帧对应特征点的平均视差大于30，且内点数目大于12的帧，此时可进行初始化，同时返回当前帧到第l帧的坐标系变换R和T
 * */

bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
        //寻找第i帧到窗口最后一帧的对应特征点，存放在corres中
        corres = getCorresponding(i, WINDOW_SIZE);
        //匹配的特征点对要大于20
        if (corres.size() > 20)
        {
            //计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                 //第j个对应点在第i帧和最后一帧的(x,y)
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                //计算视差
                double parallax = (pts_0 - pts_1).norm();
                //计算视差的总和
                sum_parallax = sum_parallax + parallax;
            }
            //计算平视差
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //判断是否满足初始化条件：视差>30和内点数满足要求(大于12)
            //solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R，relative_T
            std::cout << " i  : "  << i << " average_parallax goal : " << average_parallax * 702 << std::endl;
            if (average_parallax * 460 > 30 && solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}


bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image)
{
    //ROS_DEBUG("input feature: %d", (int)image.size());
    //ROS_DEBUG("num of feature: %d", getFeatureCount());
    // 用于记录所有特征点的视差总和
    double parallax_sum = 0;
    // 记录满足某些条件的特征点个数
    int parallax_num = 0;
    // 被跟踪点的个数
    int last_track_num = 0;
    //遍历图像image中所有的特征点，和已经记录了特征点的容器feature中进行比较
    
    for (auto &id_pts : image)
    {
        //特征点管理器，存储特征点格式：首先按照特征点ID，一个一个存储，每个ID会包含其在不同帧上的位置
        //这里id_pts.second[0].second获取的信息为：xyz_uv_velocity << x, y, z, p_u, p_v, 0, 0
        FeaturePerFrame f_per_fra(id_pts.second[0].second);
        //获取feature_id
        int feature_id = id_pts.first;
        
        //在feature中查找该feature_id的feature是否存在
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature.end())
        {
            //没有找到该feature的id，则把特征点放入feature的list容器中
            feature.push_back(FeaturePerId(feature_id, frame_count));
            //feature是个list类型，里边每个元素类型为FeaturePerId，feature_per_frame表示每个FeaturePerId类型元素
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
             /**
             * 如果找到了相同ID特征点，就在其FeaturePerFrame内增加此特征点在此帧的位置以及其他信息，
             * it的feature_per_frame容器中存放的是该feature能够被哪些帧看到，存放的是在这些帧中该特征点的信息
             * 所以，feature_per_frame.size的大小就表示有多少个帧可以看到该特征点
             * */
            it->feature_per_frame.push_back(f_per_fra);
            //这张图片有多少个特征点被成功追踪
            last_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        //倒数第二帧与倒数第三帧
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        // std::cout << " parallax_sum / parallax_num : " << parallax_sum / parallax_num << std::endl;
        // std::cout << "MIN_PARALLAX :  " << MIN_PARALLAX << std::endl;
        return parallax_sum / parallax_num >= 10;
    }
}

void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image){

    addFeatureCheckParallax(frame_count,image);

    if(frame_count == WINDOW_SIZE){
        //将全局feature当中的特征存储到sfm_f中
        // global sfm
        Eigen::Quaterniond Q[frame_count + 1];
        Eigen::Vector3d T[frame_count + 1];
        std::map<int, Eigen::Vector3d> sfm_tracked_points;
        std::vector<SFMFeature> sfm_f;
        //feature中的feature存储到sfm_f中
        for (auto &it_per_id : feature)
        {
            int img_j = it_per_id.start_frame - 1;
            SFMFeature tmp_feature;
            tmp_feature.state = false;
            tmp_feature.id = it_per_id.feature_id;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                img_j++;    
                Eigen::Vector3d pts_j = it_per_frame.point;
                //遍历每一个能观察到该feature的frame
                tmp_feature.observation.push_back(std::make_pair(img_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
            }
            sfm_f.push_back(tmp_feature);
        }
        //遍历feature，获得里边每一个特征点对应的图像帧中点的x,y坐标存入tmp_feature中，
        //最后将tmp_feature存入到sfm_f中。后边就可以通过sfm_f获取到feature的id以及有哪些帧可以观测到该feature以及在图像帧中的坐标。
        //位姿求解
        Eigen::Matrix3d relative_R;
        Eigen::Vector3d relative_T;
        int l;
        //通过求取本质矩阵来求解出位姿
        /**
         * 这里的l表示滑动窗口中第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，
         * 会作为参考帧到下面的全局sfm使用，得到的Rt为最后一帧到第l帧的坐标系变换Rt，存储在relative_R和relative_T当中
         * */

        if (!relativePose(relative_R, relative_T, l))
        {
            std::cout << "Not enough features or parallax; Move device around" << std::endl;
            return ;
        }

        //三角化求解地图点的深度
        GlobalSFM sfm;
        //三角化特征点，对滑窗每一帧求解sfm问题 
        //为了解决之前得到的相对位姿的尺度未知问题 但其实还不够准确，后面结合imu对齐得到更为精确的尺度
        sfm.construct(frame_count + 1, Q, T, l,
                            relative_R, relative_T,
                            sfm_f, sfm_tracked_points);
    }
    else{
        frame_count++;
    }
}

int main(int argv, char ** argc) {

    std::string img_dir_path = argc[1];
    const int img_num = atoi(argc[2]);
    
    std::vector<cv::Mat> sfm_img_;
    FeatureTraker feature_track;
    // for()

    for(int i = 0; i < img_num; ++i){
        
        const std::string img_path = img_dir_path + '/' + std::to_string(i) + ".jpeg";
        std::cout << "img_path : " << img_path << std::endl;
        cv::Mat img_ = cv::imread(img_path,0);
        COL = img_.cols;
        ROW = img_.rows;
        std::cout << " COL : " << COL << std::endl;
        std::cout << " ROW : " << ROW << std::endl;
        // sfm_img_.push_back(img_);
        feature_track.readImage(img_);

        for (unsigned int i = 0;; i++)
        {
            bool completed = false;
            completed |= feature_track.updateID(i); //

            if (!completed)
                break;
        }


        // cv::imshow("img",img_);
        // cv::waitKey(0);
        if (PUB_THIS_FRAME)
        {
            std::shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
            const int NUM_OF_CAM = 1;
            std::vector<std::set<int>> hash_ids(NUM_OF_CAM);
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                auto &un_pts = feature_track.cur_un_pts;
                auto &cur_pts = feature_track.cur_pts;
                auto &ids = feature_track.ids;

                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if (feature_track.track_cnt[j] > 1)
                    {
                        int p_id = ids[j];
                        hash_ids[i].insert(p_id);
                        double x = un_pts[j].x;
                        double y = un_pts[j].y;
                        double z = 1;
                        feature_points->points.push_back(Eigen::Vector3d(x, y, z));
                        feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                        feature_points->u_of_point.push_back(cur_pts[j].x);
                        feature_points->v_of_point.push_back(cur_pts[j].y);
                       
                    }
                }
                //}
                // skip the first image; since no optical speed on frist image
                if (!init_pub)
                {
                    std::cout << "4 PubImage init_pub skip the first image!" << std::endl;
                    init_pub = 1;
                }
                else
                {
                    feature_buf.push(feature_points);
                    // cout << "5 PubImage t : " << fixed << feature_points->header
                    //     << " feature_buf size: " << feature_buf.size() << endl;
                }
            }
        }

        if(!feature_buf.empty()){
            ImgConstPtr img_msg = feature_buf.front();

            std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;

            for (unsigned int i = 0; i < img_msg->points.size(); i++) 
            {
                //id_of_point 特征点的编号
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / 1; //
                int camera_id = v % 1;
                // std::cout << "camera_id" << camera_id << std::endl;
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, 0.0, 0.0;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            processImage(image);
        }
                    // show tacker points
        cv::Mat show_img;
        cv::cvtColor(img_, show_img, CV_GRAY2RGB);
        
        for (unsigned int j = 0; j < feature_track.cur_pts.size(); j++)
        {
            double len = std::min(1.0, 1.0 * feature_track.track_cnt[j] / 30);
            cv::circle(show_img, feature_track.cur_pts[j], 10, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }

        cv::namedWindow("IMAGE", cv::WINDOW_FREERATIO);
        cv::imshow("IMAGE", show_img);
        cv::waitKey(0);
        // std::cout << "img_dir_path : " << img_dir_path << std::endl;
        // std::cout << "img_num : " << img_num << std::endl;

    }
    
    return 0;

}