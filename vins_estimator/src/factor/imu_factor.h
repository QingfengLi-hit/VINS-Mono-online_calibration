#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>

class IMUFactor : public ceres::SizedCostFunction<18, 7, 9, 7, 9, 3, 3>
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        Eigen::Vector3d Eiai(parameters[4][0], parameters[4][1], parameters[4][2]);
        Eigen::Vector3d Eiaj(parameters[5][0], parameters[5][1], parameters[5][2]);


#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        Eigen::Map<Eigen::Matrix<double, 18, 1>> residual(residuals);

        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                             Pj, Qj, Vj, Baj, Bgj,
                                             Eiai, Eiaj);

        //std::cout<<"residual= "<<residual<<std::endl;
        Eigen::Matrix<double, 18, 18> sqrt_info;
        sqrt_info = Eigen::LLT<Eigen::Matrix<double, 18, 18>>(pre_integration->covariance.inverse()).matrixL().transpose();
        //std::cout<<" pre_integration->covariance = "<<pre_integration->covariance<<std::endl;
        //std::cout<<" sqrt_info = "<<sqrt_info<<std::endl;
        residual = sqrt_info * residual;
        //std::cout<<" residual = "<<residual<<std::endl;
        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);


            Eigen::Matrix3d dv_dQia = pre_integration->jacobian.template block<3, 3>(O_V, 15);
            Eigen::Matrix3d dp_dQia = pre_integration->jacobian.template block<3, 3>(O_P, 15);


            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }
            Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));

            Eigen::Matrix3d Ria = pre_integration->Ria;
            Utility utility;
            Quaterniond Qiai = utility.euler_to_quat(Eiai);
            Quaterniond Qiaj = utility.euler_to_quat(Eiaj);
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(O_P, 0) = -Qi.inverse().toRotationMatrix();

                jacobian_pose_i.block<3, 3>(O_P, 3) = Utility::skewSymmetric(Qi.inverse() * (0.5 * Ria *  G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
#if 0
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                //corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, 3) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif
                jacobian_pose_i.block<3, 3>(O_V, 3) = Utility::skewSymmetric(Qi.inverse() * (Ria * G * sum_dt + Vj - Vi));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 18, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(O_P, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, 3) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, 6) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else
                //corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_speedbias_i.block<3, 3>(O_R, 6) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(O_V, 0) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, 3) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, 6) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(O_BA,3) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(O_BG, 6) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(O_P, 0) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else
                //corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_j.block<3, 3>(O_R, 3) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
               // ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 18, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(O_V, 0) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(O_BA, 3) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(O_BG, 6) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }

            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>> jacobian_imu_Eiai(jacobians[4]);
                jacobian_imu_Eiai.setZero();
                jacobian_imu_Eiai.block<3, 3>(O_P, 0) =  -dp_dQia;

                jacobian_imu_Eiai.block<3, 3>(O_V, 0) =  -dv_dQia;

                jacobian_imu_Eiai.block<3, 3>(15, 0) =  -(Utility::Qleft(Qiaj.inverse() * Qiai)).bottomRightCorner<3, 3>();

                jacobian_imu_Eiai = sqrt_info * jacobian_imu_Eiai;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }

            if(jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>> jacobian_imu_Eiaj(jacobians[5]);
                jacobian_imu_Eiaj.setZero();

                jacobian_imu_Eiaj.block<3, 3>(15, 0) =Utility::Qleft(Qiai.inverse() * Qiaj).bottomRightCorner<3, 3>();

                jacobian_imu_Eiaj = sqrt_info * jacobian_imu_Eiaj;

                //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }

        }
        //std::cout<<" jacobians = "<<*jacobians<<std::endl;
        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 30, 1> &residuals, Eigen::Matrix<double, 30, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
IntegrationBase* pre_integration;

};



