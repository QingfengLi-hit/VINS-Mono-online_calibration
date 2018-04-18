#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

class IntegrationBase
{
  public:
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg,
                    const Eigen::Vector3d &_linearized_Eia)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},linearized_Eia{_linearized_Eia},
          jacobian{Eigen::Matrix<double, 18, 18>::Identity()}, covariance{Eigen::Matrix<double, 18, 18>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
       noise = Eigen::Matrix<double, 15, 15>::Zero();
       noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
       noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
       //noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
       //noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
       noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
       noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

       noise.block<3, 3>(12, 12) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.Zero();
        //Eigen::Vector3d Qai = Vector3d(0,0,0);
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    Quaterniond euler_to_quat(const Vector3d YPR)
        {
            Matrix3d matYaw(3, 3), matRoll(3, 3), matPitch(3, 3), matRotation(3, 3);
            const auto yaw = YPR[2]*M_PI / 180;
            const auto pitch = YPR[0]*M_PI / 180;
            const auto roll = YPR[1]*M_PI / 180;

            matYaw << cos(yaw), sin(yaw), 0.0f,
                -sin(yaw), cos(yaw), 0.0f,  //z
                0.0f, 0.0f, 1.0f;

            matPitch << cos(pitch), 0.0f, -sin(pitch),
                0.0f, 1.0f, 0.0f,   // y
                sin(pitch), 0.0f, cos(pitch);

            matRoll << 1.0f, 0.0f, 0.0f,
                0.0f, cos(roll), sin(roll),   // x
                0.0f, -sin(roll), cos(roll);

            matRotation = matYaw*matPitch*matRoll;

            Quaterniond quatFromRot(matRotation);

            quatFromRot.normalize(); //Do i need to do this?

            return quatFromRot;
        }

       Vector3d quat_to_euler(const Eigen::Quaterniond& q)
       {
           Vector3d retVector;

           const auto x = q.y();
           const auto y = q.z();
           const auto z = q.x();
           const auto w = q.w();

           retVector[2] = atan2(2.0 * (y * z + w * x), w * w - x * x - y * y + z * z);
           retVector[1] = asin(-2.0 * (x * z - w * y));
           retVector[0] = atan2(2.0 * (x * y + w * z), w * w + x * x - y * y - z * z);


           //retVector[0] = (retVector[0] * (180 / M_PI));
           //retVector[1] = (retVector[1] * (180 / M_PI))*-1;
           //retVector[2] = retVector[2] * (180 / M_PI);

           return retVector;
       }
    void midPointIntegration(double _dt,
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,                        
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            const Eigen::Vector3d &Qia_v,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, Eigen::Vector3d &result_linearized_Qia, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");

          Quaterniond Qia = euler_to_quat(Qia_v);
          //std::cout<<"Qia= "<<Qia.x()<<" "<<Qia.y()<<" "<<Qia.z()<<" "<<Qia.w()<<endl;
          Vector3d un_acc = delta_q * Qia * (_acc_0 - linearized_ba);
          Vector3d un_gyr = (_gyr_0  - linearized_bg);

          result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
          result_delta_v = delta_v + un_acc * _dt;
          result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);

          result_linearized_ba = linearized_ba;
          result_linearized_bg = linearized_bg;
          result_linearized_Qia = Qia_v;

          if(update_jacobian)
          {
              Vector3d a_0_x = _acc_0 - linearized_ba;
              Vector3d a_1_x = Qia.toRotationMatrix()*(_acc_0 - linearized_ba);
              Matrix3d R_a_0_x, R_a_1_x, R_g;

              R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                  a_0_x(2), 0, -a_0_x(0),
                  -a_0_x(1), a_0_x(0), 0;

              R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                  a_1_x(2), 0, -a_1_x(0),
                  -a_1_x(1), a_1_x(0), 0;

              R_g<<0, -un_gyr(2), un_gyr(1),
                  un_gyr(2), 0, -un_gyr(0),
                  -un_gyr(1), un_gyr(0), 0;


             MatrixXd F = MatrixXd::Zero(18, 18);
             F.block<3, 3>(0, 0) = Matrix3d::Identity();
             F.block<3, 3>(0, 3) = Matrix3d::Identity() * _dt;


             F.block<3, 3>(3, 3) = Matrix3d::Identity();
             F.block<3, 3>(3, 6) = -delta_q.toRotationMatrix() * R_a_1_x *_dt;
             F.block<3, 3>(3, 9) = -delta_q.toRotationMatrix() * Qia *_dt;
             F.block<3, 3>(3, 15)= -delta_q.toRotationMatrix() * Qia * R_a_0_x * _dt;

             Utility utility;
             Eigen::Matrix3d tmp_w = utility.ypr2R(un_gyr*_dt);
             F.block<3, 3>(6, 6) = -R_g*_dt + Matrix3d::Identity();
             F.block<3, 3>(6, 12)= -Matrix3d::Identity()*_dt;

             F.block<3, 3>(9, 9) = Matrix3d::Identity();
             F.block<3, 3>(12, 12) = Matrix3d::Identity();
             F.block<3, 3>(15, 15) = Matrix3d::Identity();

              MatrixXd V = MatrixXd::Zero(18,15);
              V.block<3, 3>(3, 0) =  -delta_q.toRotationMatrix()* Qia * _dt;

              V.block<3, 3>(6, 3) =  -Matrix3d::Identity() * _dt;
              V.block<3, 3>(9, 6) = MatrixXd::Identity(3,3) * _dt;
              V.block<3, 3>(12, 9) = MatrixXd::Identity(3,3) * _dt;
              V.block<3, 3>(15, 12) = MatrixXd::Identity(3,3) * _dt;

              //step_jacobian = F;
              //step_V = V;
              jacobian = F * jacobian;

              covariance = F * covariance * F.transpose() + V * noise * V.transpose();
              //std::cout<<" covariance ="<<covariance<<std::endl;
        }

    }

   /*void midPointIntegration(double _dt,
                                const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                                const Eigen::Vector3d &Qia_v,
                                const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                                Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                                Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
      {
            //ROS_INFO("midpoint integration");
            Quaterniond Qia = euler_to_quat(Qia_v);
            Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
            Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
            Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
            result_delta_v = delta_v + un_acc * _dt;
            result_linearized_ba = linearized_ba;
            result_linearized_bg = linearized_bg;

            if(update_jacobian)
            {
                Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
                Vector3d a_0_x = _acc_0 - linearized_ba;
                Vector3d a_1_x = _acc_1 - linearized_ba;
                Vector3d a_0q_x = Qia.toRotationMatrix()*(_acc_0 - linearized_ba);
                Vector3d a_1q_x = Qia.toRotationMatrix()*(_acc_1 - linearized_ba);
                Matrix3d R_w_x, R_a_0_x, R_a_1_x,R_a_q0_x, R_a_q1_x;

                R_w_x<<0, -w_x(2), w_x(1),
                    w_x(2), 0, -w_x(0),
                    -w_x(1), w_x(0), 0;
                R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                    a_0_x(2), 0, -a_0_x(0),
                    -a_0_x(1), a_0_x(0), 0;
                R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                    a_1_x(2), 0, -a_1_x(0),
                    -a_1_x(1), a_1_x(0), 0;
                R_a_q0_x<<0, -a_0q_x(2), a_0q_x(1),
                    a_0q_x(2), 0, -a_0q_x(0),
                    -a_0q_x(1), a_0q_x(0), 0;
                R_a_q1_x<<0, -a_1q_x(2), a_1q_x(1),
                    a_1q_x(2), 0, -a_1q_x(0),
                    -a_1q_x(1), a_1q_x(0), 0;

                MatrixXd F = MatrixXd::Zero(18, 18);
                F.block<3, 3>(0, 0) = Matrix3d::Identity();
                F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix()* Qia * R_a_0_x * _dt * _dt +
                                      -0.25 * result_delta_q.toRotationMatrix()* Qia  * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
                F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
                F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
                F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix()* Qia  *  R_a_1_x * _dt * _dt * -_dt;
                F.block<3, 3>(0, 15) = -0.25 * delta_q.toRotationMatrix()* Qia * R_a_q0_x * _dt * _dt +
                                       -0.25 * result_delta_q.toRotationMatrix()* Qia  * R_a_q1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;

                F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
                F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
                F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix()* Qia  * R_a_0_x * _dt +
                                      -0.5 * result_delta_q.toRotationMatrix()* Qia  * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
                F.block<3, 3>(6, 6) = Matrix3d::Identity();
                F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
                F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix()* Qia  * R_a_1_x * _dt * -_dt;
                F.block<3, 3>(6, 15) = -0.5 * delta_q.toRotationMatrix()* Qia  * R_a_q0_x * _dt +
                                       -0.5 * result_delta_q.toRotationMatrix()* Qia  * R_a_q1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;

                F.block<3, 3>(9, 9) = Matrix3d::Identity();
                F.block<3, 3>(12, 12) = Matrix3d::Identity();
                F.block<3, 3>(15, 15) = Matrix3d::Identity();
                //cout<<"A"<<endl<<A<<endl;

                MatrixXd V = MatrixXd::Zero(18,18);
                V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
                V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix()* Qia  * R_a_1_x  * _dt * _dt * 0.5 * _dt;
                V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
                V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
                V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
                V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
                V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
                V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * Qia * R_a_1_x  * _dt * 0.5 * _dt;
                V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
                V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
                V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
                V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

                //step_jacobian = F;
                //step_V = V;
                jacobian = F * jacobian;
                covariance = F * covariance * F.transpose() + V * noise * V.transpose();
            }

        }*/

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
       {
              dt = _dt;
              acc_1 = _acc_1;
              gyr_1 = _gyr_1;
              Vector3d result_delta_p;
              Quaterniond result_delta_q;
              Vector3d result_delta_v;
              Vector3d result_linearized_ba;
              Vector3d result_linearized_bg;
              Vector3d result_linearized_Qai;
              midPointIntegration(_dt,
                                  acc_0, gyr_0,
                                  //_acc_1, _gyr_1,
                                  delta_p, delta_q, delta_v, linearized_ba, linearized_bg,linearized_Eia,
                                  result_delta_p, result_delta_q, result_delta_v,
                                  result_linearized_ba, result_linearized_bg, result_linearized_Qai, 1);

              //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
              //                    linearized_ba, linearized_bg);
              delta_p = result_delta_p;
              delta_q = result_delta_q;
              delta_v = result_delta_v;
              linearized_ba = result_linearized_ba;
              linearized_bg = result_linearized_bg;
              linearized_Eia = result_linearized_Qai;
              delta_q.normalize();
              sum_dt += dt;
              acc_0 = acc_1;
              gyr_0 = gyr_1;

       }


    Eigen::Matrix<double, 18, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj,         
                                          const Eigen::Vector3d &Eiai,const Eigen::Vector3d &Eiaj)
    {

        Utility utility;
        Ria = utility.ypr2R(linearized_Eia);

        Eigen::Matrix<double, 18, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Matrix3d dv_dQia = jacobian.block<3, 3>(O_V, 15);
        Eigen::Matrix3d dp_dQia = jacobian.block<3, 3>(O_P, 15);

        Eigen::Vector3d dQia = Eiai - linearized_Eia;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);

        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba *  dba + dv_dbg * dbg
                                          + dv_dQia * dQia;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba *  dba + dp_dbg * dbg
                                          + dp_dQia * dQia;
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5  * Ria* G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (Ria * G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        Quaterniond Qiai,Qiaj;
        Qiai = euler_to_quat(Eiai);
        Qiaj = euler_to_quat(Eiaj);
        residuals.block<3, 1>(15, 0) = 2*(Qiai.inverse() * Qiaj).vec() ;


        /*double x = _Qai.x();
        double y = _Qai.y();
        double z = _Qai.z();
        double gx = G.x();
        double gy = G.y();
        double gz = G.z();

        J_Qai(0,0) = (cos(z)*sin(y)*cos(x)+sin(z)*sin(x)*gy)-(cos(z)*sin(y)*sin(x)+sin(z)*cos(x))*gz;
        J_Qai(0,1) = (-sin(y)*cos(z))*gx + (cos(z)*cos(y)*sin(x))*gy+ (cos(z)*cos(y)*cos(x))*gz;
        J_Qai(0,2) = (-cos(y)*sin(z))*gx + (-sin(z)*sin(y)*sin(x)-cos(z)*cos(x))*gy + (-sin(z)*sin(y)*cos(x)+cos(z)*sin(x))*gz;

        J_Qai(1,0) = (sin(z)*sin(y)*cos(x)-cos(z)*sin(x))*gy + (-sin(z)*sin(y)*sin(x)-cos(z)*cos(x))*gz;
        J_Qai(1,1) = -sin(y)*sin(z)*gx + sin(z)*cos(y)*sin(x)*gy + sin(z)*cos(y)*cos(x)*gz;
        J_Qai(1,2) = cos(y)*cos(z)*gx + (cos(z)*sin(y)*sin(x)-sin(z)*cos(x))*gy + (cos(z)*sin(y)*cos(x)+sin(z)*sin(x))*gz;

        J_Qai(2,0) = cos(x)*cos(y)*gy-sin(x)*sin(y)*gz;
        J_Qai(2,1) = -cos(y)*gx-sin(x)*sin(y)*gy+cos(x)*cos(y)*gz;
        J_Qai(2,2) = 0;

        J_P_Qai = 0.5 * sum_dt * sum_dt * Qi.inverse().toRotationMatrix() *J_Qai - dp_dQia;
        J_V_Qai = sum_dt *  Qi.inverse().toRotationMatrix() *J_Qai - dv_dQia;*/
        return residuals;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg, linearized_Eia;

    Eigen::Matrix<double, 18, 18> jacobian, covariance;

    Eigen::Matrix<double, 15, 15> step_jacobian;

    Eigen::Matrix<double, 15, 15> noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf,dt_buf_imuJ;
    std::vector<Eigen::Vector3d> acc_buf,acc_buf_imuJ;
    std::vector<Eigen::Vector3d> gyr_buf,gyr_buf_imuJ;

    Eigen::Vector3d _Ka;
    Eigen::Vector3d _Ma;
    Eigen::Vector3d _Kg;
    Eigen::Vector3d _Mg;
    Eigen::Vector3d _Qai;

    double __dt;
    double _sum_dt;
    Eigen::Vector3d _delta_p;
    Eigen::Quaterniond _delta_q;
    Eigen::Vector3d _delta_v;

    Eigen::Matrix3d Ria;
    Eigen::Matrix3d J_Qai;
    Eigen::Matrix3d J_P_Qai;
    Eigen::Matrix3d J_V_Qai;

};

/*

    void eulerIntegration(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        result_delta_p = delta_p + delta_v * _dt + 0.5 * (delta_q * (_acc_1 - linearized_ba)) * _dt * _dt;
        result_delta_v = delta_v + delta_q * (_acc_1 - linearized_ba) * _dt;
        Vector3d omg = _gyr_1 - linearized_bg;
        omg = omg * _dt / 2;
        Quaterniond dR(1, omg(0), omg(1), omg(2));
        result_delta_q = (delta_q * dR);   
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = _gyr_1 - linearized_bg;
            Vector3d a_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_x<<0, -a_x(2), a_x(1),
                a_x(2), 0, -a_x(0),
                -a_x(1), a_x(0), 0;

            MatrixXd A = MatrixXd::Zero(15, 15);
            // one step euler 0.5
            A.block<3, 3>(0, 3) = 0.5 * (-1 * delta_q.toRotationMatrix()) * R_a_x * _dt;
            A.block<3, 3>(0, 6) = MatrixXd::Identity(3,3);
            A.block<3, 3>(0, 9) = 0.5 * (-1 * delta_q.toRotationMatrix()) * _dt;
            A.block<3, 3>(3, 3) = -R_w_x;
            A.block<3, 3>(3, 12) = -1 * MatrixXd::Identity(3,3);
            A.block<3, 3>(6, 3) = (-1 * delta_q.toRotationMatrix()) * R_a_x;
            A.block<3, 3>(6, 9) = (-1 * delta_q.toRotationMatrix());
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd U = MatrixXd::Zero(15,12);
            U.block<3, 3>(0, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            U.block<3, 3>(3, 3) =  MatrixXd::Identity(3,3);
            U.block<3, 3>(6, 0) =  delta_q.toRotationMatrix();
            U.block<3, 3>(9, 6) = MatrixXd::Identity(3,3);
            U.block<3, 3>(12, 9) = MatrixXd::Identity(3,3);

            // put outside
            Eigen::Matrix<double, 12, 12> noise = Eigen::Matrix<double, 12, 12>::Zero();
            noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

            //write F directly
            MatrixXd F, V;
            F = (MatrixXd::Identity(15,15) + _dt * A);
            V = _dt * U;
            step_jacobian = F;
            step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }     


    void checkJacobian(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0, 
                                   const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
    {
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 0);

        Vector3d turb_delta_p;
        Quaterniond turb_delta_q;
        Vector3d turb_delta_v;
        Vector3d turb_linearized_ba;
        Vector3d turb_linearized_bg;

        Vector3d turb(0.0001, -0.003, 0.003);

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p + turb, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb p       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff " << (step_jacobian.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q * Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2), delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb q       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v + turb,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb v       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba + turb, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb ba       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 9) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg + turb,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb bg       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 12) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 12) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 12) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 12) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 12) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0 + turb, _gyr_0, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0 + turb, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 + turb, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 , _gyr_1 + turb, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 9) * turb).transpose() << endl;
    }
    */
