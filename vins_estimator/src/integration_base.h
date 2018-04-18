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
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
            //dt_buf_imuJ{dt_buf_imuJ.clear()},acc_buf_imuJ{acc_buf_imuJ.clear()},gyr_buf_imuJ{gyr_buf_imuJ.clear()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
        dt_buf_imuJ.clear();
        acc_buf_imuJ.clear();
        gyr_buf_imuJ.clear();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr,
                   const Eigen::Vector3d &Ka, const Eigen::Vector3d &Ma,
                   const Eigen::Vector3d &Kg, const Eigen::Vector3d &Mg,
                   const Eigen::Vector3d &Qai)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);

        dt_buf_imuJ.push_back(dt);
        acc_buf_imuJ.push_back(acc);
        gyr_buf_imuJ.push_back(gyr);

        _Ka = Ka;
        _Ma = Ma;
        _Kg = Kg;
        _Mg = Mg;
        _Qai= Qai;
        propagate(dt, acc, gyr, Ka, Ma, Kg, Mg, Qai,false);
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
        covariance.setZero();
        Eigen::Vector3d Ka = Vector3d(1,1,1);
        Eigen::Vector3d Ma = Vector3d(0,0,0);
        Eigen::Vector3d Kg = Vector3d(1,1,1);
        Eigen::Vector3d Mg = Vector3d(0,0,0);
        Eigen::Vector3d Qai = Vector3d(0,0,0);
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i], Ka, Ma, Kg, Mg, Qai,false);
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
            0.0f, 1.0f, 0.0f,   // X
            sin(pitch), 0.0f, cos(pitch);

        matRoll << 1.0f, 0.0f, 0.0f,
            0.0f, cos(roll), sin(roll),   // Y
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
    void midPointIntegration(double _dt, bool get_imu_intr_J,
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                            const Eigen::Vector3d &Kg,  const Eigen::Vector3d &Mg,
                            const Eigen::Vector3d &Qai,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");

        //Get imu model parameter
    Eigen::Matrix3d Ta,Tg;
    Ta(0,0) = Ka.x(); Ta(0,1) = Ma.x(); Ta(0,2) = Ma.y();
    Ta(1,0) = 0;      Ta(1,1) = Ka.y(); Ta(1,2) = Ma.z();
    Ta(2,0) = 0;      Ta(2,1) = 0;      Ta(2,2) = Ka.z();

    Tg(0,0) = Kg.x(); Tg(0,1) = Mg.x(); Tg(0,2) = Mg.y();
    Tg(1,0) = 0;      Tg(1,1) = Kg.y(); Tg(1,2) = Mg.z();
    Tg(2,0) = 0;      Tg(2,1) = 0;      Tg(2,2) = Kg.z();

    Eigen::Matrix3d Ta_inv,Tg_inv;
    Eigen::Quaterniond Qai_inv;
    Ta_inv = Ta.inverse();
    Tg_inv = Tg.inverse();
    Quaterniond tmp_q = euler_to_quat(Qai);
    Qai_inv = tmp_q.inverse();
    //std::cout<<"Ta_inv= "<<Ta_inv<<std::endl;
    //std::cout<<"Tg_inv= "<<Tg_inv<<std::endl;
    //std::cout<<"Qai_inv= "<<Qai_inv.x()<<Qai_inv.y()<<Qai_inv.z()<<Qai_inv.w()<<std::endl;
    Vector3d un_acc_0 = delta_q * Qai_inv * Ta_inv *  (_acc_0 - linearized_ba);
    Vector3d un_gyr = Tg_inv * (0.5 * (_gyr_0 + _gyr_1) - linearized_bg);
    result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
    Vector3d un_acc_1 = result_delta_q * Qai_inv * Ta_inv * (_acc_1 - linearized_ba);
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    result_delta_v = delta_v + un_acc * _dt;
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    if(update_jacobian&&!get_imu_intr_J)
    {
        Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        Vector3d a_0_x = _acc_0 - linearized_ba;
        Vector3d a_1_x = _acc_1 - linearized_ba;
        Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x<<0, -w_x(2), w_x(1),
            w_x(2), 0, -w_x(0),
            -w_x(1), w_x(0), 0;
        R_a_0_x<<0, -a_0_x(2), a_0_x(1),
            a_0_x(2), 0, -a_0_x(0),
            -a_0_x(1), a_0_x(0), 0;
        R_a_1_x<<0, -a_1_x(2), a_1_x(1),
            a_1_x(2), 0, -a_1_x(0),
            -a_1_x(1), a_1_x(0), 0;

           MatrixXd F = MatrixXd::Zero(15, 15);
           F.block<3, 3>(0, 0) = Matrix3d::Identity();
           F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * Qai_inv.toRotationMatrix() * Ta_inv *  R_a_0_x * _dt * _dt +
                                 -0.25 * result_delta_q.toRotationMatrix() * Qai_inv.toRotationMatrix() * Ta_inv * R_a_1_x * (Matrix3d::Identity() - Tg_inv * R_w_x * _dt) * _dt * _dt;
           F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
           F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() * Qai_inv.toRotationMatrix() * Ta_inv + result_delta_q.toRotationMatrix()* Qai_inv.toRotationMatrix() * Ta_inv) * _dt * _dt;
           F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * Qai_inv.toRotationMatrix() * Ta_inv * R_a_1_x * _dt * _dt * -_dt;


           F.block<3, 3>(3, 3) = Matrix3d::Identity() - Tg_inv*R_w_x * _dt;
           F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;

           F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * Qai_inv.toRotationMatrix() * Ta_inv * R_a_0_x * _dt +
                                 -0.5 * result_delta_q.toRotationMatrix() * Qai_inv.toRotationMatrix() * Ta_inv * R_a_1_x * (Matrix3d::Identity() -Tg_inv * R_w_x * _dt) * _dt;
           F.block<3, 3>(6, 6) = Matrix3d::Identity();
           F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix()*Qai_inv.toRotationMatrix() * Ta_inv + result_delta_q.toRotationMatrix()*Qai_inv.toRotationMatrix() * Ta_inv) * _dt;
           F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() *Qai_inv.toRotationMatrix() * Ta_inv* R_a_1_x * _dt * -_dt;


           F.block<3, 3>(9, 9) = Matrix3d::Identity();
           F.block<3, 3>(12, 12) = Matrix3d::Identity();


        MatrixXd V = MatrixXd::Zero(15,18);
        V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
        V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
        V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
        V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
        V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

        //step_jacobian = F;
        //step_V = V;
        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();

    }

    }

    Eigen::Matrix<double, 9, 15> Get_imu_intr_J(const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                        const Eigen::Vector3d &Kg,const Eigen::Vector3d &Mg,const Eigen::Vector3d &Qai)
    {
      Eigen::Matrix<double, 9, 15>  jacobian;
      jacobian.block<9,3>(0,0) = Get_block_J(Ka,0,Ka, Ma, Kg, Mg, Qai);
      //std::cout<<"Ka=  "<<jacobian.block<9,3>(0,0)<<std::endl;
      jacobian.block<9,3>(0,3) = Get_block_J(Ma,1,Ka, Ma, Kg, Mg, Qai);
      //std::cout<<"Ma=  "<<jacobian.block<9,3>(0,3)<<std::endl;
      jacobian.block<9,3>(0,6) = Get_block_J(Kg,2,Ka, Ma, Kg, Mg, Qai);
      //std::cout<<"Kg=  "<<jacobian.block<9,3>(0,6)<<std::endl;
      jacobian.block<9,3>(0,9) = Get_block_J(Mg,3,Ka, Ma, Kg, Mg, Qai);
      //std::cout<<"Mg=  "<<jacobian.block<9,3>(0,9)<<std::endl;
      jacobian.block<9,3>(0,12)= Get_block_J(Qai,4,Ka, Ma, Kg, Mg, Qai);
      //std::cout<<"Qai=  "<<jacobian.block<9,3>(0,12)<<std::endl;

      //std::cout<<"jacobian=  "<<jacobian<<std::endl;
      return jacobian;
      _covariance = jacobian.transpose() * jacobian;
    }

    Eigen::Matrix<double, 9, 3> Get_block_J(const Eigen::Vector3d &variate, const int m,
                             const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                             const Eigen::Vector3d &Kg,const Eigen::Vector3d &Mg,const Eigen::Vector3d &Qai)
    {
        const double delta = 1e-12;
        const double scalar = 1.0 / (2*delta);
        Eigen::Matrix<double, 9, 3>  _jacobian_block;
        Eigen::Vector3d variate_front = variate;
        Eigen::Vector3d variate_after = variate;
        Eigen::Matrix<double, 9, 1> front;
        Eigen::Matrix<double, 9, 1> after;
        for(int i=0; i<3; i++)
        {
            //compute delta
            if(i==0)
            {
               variate_front.x() = variate.x() - delta;
               variate_after.x() = variate.x() + delta;
            }
            if(i==1)
            {
               variate_front.y() = variate.y() - delta;
               variate_after.y() = variate.y() + delta;
            }
            if(i==2)
            {
               variate_front.z() = variate.z() - delta;
               variate_after.z() = variate.z() + delta;
            }
            imu_propagate(variate_front,m,Ka, Ma, Kg, Mg, Qai);
            front.block<3,1>(0,0) = delta_p;
            front.block<3,1>(3,0) = quat_to_euler(delta_q);
            front.block<3,1>(6,0) = delta_v;
            imu_propagate(variate_after,m,Ka, Ma, Kg, Mg, Qai);
            after.block<3,1>(0,0) = delta_p;
            after.block<3,1>(3,0) = quat_to_euler(delta_q);
            after.block<3,1>(6,0) = delta_v;
          _jacobian_block.col(i) = scalar * (after - front);
        }
       return _jacobian_block;
    }

    void imu_propagate(const Eigen::Vector3d &variate, const int m,
                       const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                       const Eigen::Vector3d &Kg,const Eigen::Vector3d &Mg,const Eigen::Vector3d &Qai)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        //jacobian.setIdentity();
        //covariance.setZero();
       // std::cout<<"dt_buf_imuJ.size()= "<<dt_buf_imuJ.size()<<std::endl;
        if(m==0)
        for (int i = 0; i < static_cast<int>(dt_buf_imuJ.size()); i++)
            propagate(dt_buf_imuJ[i], acc_buf_imuJ[i], gyr_buf_imuJ[i], variate, Ma, Kg, Mg, Qai,true);
        if(m==1)
        for (int i = 0; i < static_cast<int>(dt_buf_imuJ.size()); i++)
            propagate(dt_buf_imuJ[i], acc_buf_imuJ[i], gyr_buf_imuJ[i], Ka, variate, Kg, Mg, Qai,true);
        if(m==2)
        for (int i = 0; i < static_cast<int>(dt_buf_imuJ.size()); i++)
            propagate(dt_buf_imuJ[i], acc_buf_imuJ[i], gyr_buf_imuJ[i], Ka, Ma, variate, Mg, Qai,true);
        if(m==3)
        for (int i = 0; i < static_cast<int>(dt_buf_imuJ.size()); i++)
            propagate(dt_buf_imuJ[i], acc_buf_imuJ[i], gyr_buf_imuJ[i], Ka, Ma, Kg, variate, Qai,true);
        if(m==4)
        for (int i = 0; i < static_cast<int>(dt_buf_imuJ.size()); i++)
            propagate(dt_buf_imuJ[i], acc_buf_imuJ[i], gyr_buf_imuJ[i], Ka, Ma, Kg, Mg, variate,true);

    }

    /*Eigen::Matrix3d Get_J_block(const Eigen::Vector3d &variate, const int m, const int n,
                                    const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                                    const Eigen::Vector3d &Kg,const Eigen::Vector3d &Mg,const Eigen::Vector3d &Qai)
   {
    const double delta = 1e-9;
    const double scalar = 1.0 / (2*delta);
    Eigen::Matrix3d _jacobian_block;
    Vector3d variate_front = variate;
    Vector3d variate_after = variate;
    Eigen::Vector3d front;
    Eigen::Vector3d after;

    for(int i=0; i<3; i++)
    {
        //compute delta
        if(i==0)
        {
           variate_front.x() = variate.x() - delta;
           variate_after.x() = variate.x() + delta;
        }
        if(i==1)
        {
           variate_front.y() = variate.y() - delta;
           variate_after.y() = variate.y() + delta;
        }
        if(i==2)
        {
           variate_front.z() = variate.z() - delta;
           variate_after.z() = variate.z() + delta;
        }
        //compute jacobian
        if(n==1)
        {
            if(m==1)
            {
             front = Get_P(variate_front, Ma, Kg, Mg, Qai);
             after = Get_P(variate_after, Ma, Kg, Mg, Qai);
            }
            if(m==2)
            {
              front = Get_Q(variate_front, Ma, Kg, Mg, Qai);
              after = Get_Q(variate_after, Ma, Kg, Mg, Qai);
            }
            if(m==3)
            {
              front = Get_V(variate_front, Ma, Kg, Mg, Qai);
              after = Get_V(variate_after, Ma, Kg, Mg, Qai);
            }

        }
        if(n==2)
        {
            if(m==1)
            {
              front = Get_P(Ka, variate_front, Kg, Mg, Qai);
              after = Get_P(Ka, variate_after, Kg, Mg, Qai);
            }
            if(m==2)
            {
              front = Get_Q(Ka, variate_front, Kg, Mg, Qai);
              after = Get_Q(Ka, variate_after, Kg, Mg, Qai);
            }
            if(m==3)
            {
              front = Get_V(Ka, variate_front, Kg, Mg, Qai);
              after = Get_V(Ka, variate_after, Kg, Mg, Qai);
            }
        }
        if(n==3)
        {
            if(m==1)
            {
             front = Get_P(Ka, Ma, variate_front, Mg, Qai);
             after = Get_P(Ka, Ma, variate_after, Mg, Qai);
            }

            if(m==2)
            {
             front = Get_Q(Ka, Ma, variate_front, Mg, Qai);
             after = Get_Q(Ka, Ma, variate_after, Mg, Qai);
            }
            if(m==3)
            {
             front = Get_V(Ka, Ma, variate_front, Mg, Qai);
             after = Get_V(Ka, Ma, variate_after, Mg, Qai);
            }

        }
        if(n==4)
        {
            if(m==1)
            {
             front = Get_P(Ka, Ma, Kg, variate_front, Qai);
             after = Get_P(Ka, Ma, Kg, variate_after, Qai);
            }
            if(m==2)
            {
             front = Get_Q(Ka, Ma, Kg, variate_front, Qai);
             after = Get_Q(Ka, Ma, Kg, variate_after, Qai);
            }
            if(m==3)
            {
             front = Get_V(Ka, Ma, Kg, variate_front, Qai);
             after = Get_V(Ka, Ma, Kg, variate_after, Qai);
            }
        }
        if(n==5)
        {
            if(m==1)
            {
             front = Get_P(Ka, Ma, Kg, Mg, variate_front);
             after = Get_P(Ka, Ma, Kg, Mg, variate_after);
            }
            if(m==2)
            {
             front = Get_Q(Ka, Ma, Kg, Mg, variate_front);
             after = Get_Q(Ka, Ma, Kg, Mg, variate_after);
            }
            if(m==3)
            {
             front = Get_V(Ka, Ma, Kg, Mg, variate_front);
             after = Get_V(Ka, Ma, Kg, Mg, variate_after);
            }

        }
        _jacobian_block.col(i) = scalar * (after - front);
    }
   return _jacobian_block;
  }

      Eigen::Vector3d Get_P(const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                              const Eigen::Vector3d &Kg,  const Eigen::Vector3d &Mg,
                              const Eigen::Vector3d &Qai)
   {
            //Get imu model parameter
            Eigen::Matrix3d Ta,Tg;
            Ta(0,0) = Ka.x(); Ta(0,1) = Ma.x(); Ta(0,2) = Ma.y();
            Ta(1,0) = 0;      Ta(1,1) = Ka.y(); Ta(1,2) = Ma.z();
            Ta(2,0) = 0;      Ta(2,1) = 0;      Ta(2,2) = Ka.z();

            Tg(0,0) = Kg.x(); Tg(0,1) = Mg.x(); Ta(0,2) = Mg.y();
            Tg(1,0) = 0;      Tg(1,1) = Kg.y(); Ta(1,2) = Mg.z();
            Tg(2,0) = 0;      Tg(2,1) = 0;      Ta(2,2) = Kg.z();

            Eigen::Matrix3d Ta_inv,Tg_inv;
            Eigen::Quaterniond Qai_inv;
            Ta_inv = Ta.inverse();
            Tg_inv = Tg.inverse();
            Quaterniond tmp_q = euler_to_quat(Qai);
            Qai_inv = tmp_q.inverse();
            //get P
            Vector3d un_acc_0 = delta_q * Qai_inv * Ta_inv *  (acc_0 - linearized_ba);
            Vector3d un_gyr = Tg_inv * (0.5 * (gyr_0 + gyr_1) - linearized_bg);
            Eigen::Quaterniond result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);
            Vector3d un_acc_1 = result_delta_q * Qai_inv * Ta_inv * (acc_1 - linearized_ba);
            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Vector3d result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;

            return result_delta_p;
        }
      Eigen::Vector3d Get_Q(const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                              const Eigen::Vector3d &Kg,  const Eigen::Vector3d &Mg,
                              const Eigen::Vector3d &Qai)
        {
            //Get imu model parameter
            Eigen::Matrix3d Ta,Tg;
            Ta(0,0) = Ka.x(); Ta(0,1) = Ma.x(); Ta(0,2) = Ma.y();
            Ta(1,0) = 0;      Ta(1,1) = Ka.y(); Ta(1,2) = Ma.z();
            Ta(2,0) = 0;      Ta(2,1) = 0;      Ta(2,2) = Ka.z();

            Tg(0,0) = Kg.x(); Tg(0,1) = Mg.x(); Ta(0,2) = Mg.y();
            Tg(1,0) = 0;      Tg(1,1) = Kg.y(); Ta(1,2) = Mg.z();
            Tg(2,0) = 0;      Tg(2,1) = 0;      Ta(2,2) = Kg.z();

            Eigen::Matrix3d Ta_inv,Tg_inv;
            Eigen::Quaterniond Qai_inv;
            Ta_inv = Ta.inverse();
            Tg_inv = Tg.inverse();
            Quaterniond tmp_q = euler_to_quat(Qai);
            Qai_inv = tmp_q.inverse();
            //get P
            Vector3d un_gyr = Tg_inv * (0.5 * (gyr_0 + gyr_1) - linearized_bg);
            Quaterniond result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);

            return quat_to_euler(result_delta_q);

        }
       Eigen::Vector3d Get_V(const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                              const Eigen::Vector3d &Kg,  const Eigen::Vector3d &Mg,
                              const Eigen::Vector3d &Qai)
        {
            //Get imu model parameter
            Eigen::Matrix3d Ta,Tg;
            Ta(0,0) = Ka.x(); Ta(0,1) = Ma.x(); Ta(0,2) = Ma.y();
            Ta(1,0) = 0;      Ta(1,1) = Ka.y(); Ta(1,2) = Ma.z();
            Ta(2,0) = 0;      Ta(2,1) = 0;      Ta(2,2) = Ka.z();

            Tg(0,0) = Kg.x(); Tg(0,1) = Mg.x(); Ta(0,2) = Mg.y();
            Tg(1,0) = 0;      Tg(1,1) = Kg.y(); Ta(1,2) = Mg.z();
            Tg(2,0) = 0;      Tg(2,1) = 0;      Ta(2,2) = Kg.z();

            Eigen::Matrix3d Ta_inv,Tg_inv;
            Eigen::Quaterniond Qai_inv;
            Ta_inv = Ta.inverse();
            Tg_inv = Tg.inverse();
            Quaterniond tmp_q = euler_to_quat(Qai);
            Qai_inv = tmp_q.inverse();
            //get P
            Vector3d un_acc_0 = delta_q * Qai_inv * Ta_inv *  (acc_0 - linearized_ba);
            Vector3d un_gyr = Tg_inv * (0.5 * (gyr_0 + gyr_1) - linearized_bg);

            Quaterniond result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);

            Vector3d un_acc_1 = result_delta_q * Qai_inv * Ta_inv * (acc_1 - linearized_ba);
            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Vector3d result_delta_v = delta_v + un_acc * dt;

            return result_delta_v;
        }*/

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                   const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                   const Eigen::Vector3d &Kg,  const Eigen::Vector3d &Mg,
                   const Eigen::Vector3d &Qai, bool get_imu_intr_J)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, get_imu_intr_J, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                             Ka, Ma, Kg, Mg, Qai,//scale factors and misalignment
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

   Eigen::Matrix<double, 15, 1> evaluate( const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj,
                                          const Eigen::Vector3d &Ka,  const Eigen::Vector3d &Ma,
                                          const Eigen::Vector3d &Kg,  const Eigen::Vector3d &Mg,
                                          const Eigen::Vector3d &Qai)
    {
        /*Eigen::Matrix3d Ta,Tg;
        Ta(0,0) = Ka.x(); Ta(0,1) = Ma.x(); Ta(0,2) = Ma.y();
        Ta(1,0) = 0;      Ta(1,1) = Ka.y(); Ta(1,2) = Ma.z();
        Ta(2,0) = 0;      Ta(2,1) = 0;      Ta(2,2) = Ka.z();

        Tg(0,0) = Kg.x(); Tg(0,1) = Mg.x(); Tg(0,2) = Mg.y();
        Tg(1,0) = 0;      Tg(1,1) = Kg.y(); Tg(1,2) = Mg.z();
        Tg(2,0) = 0;      Tg(2,1) = 0;      Tg(2,2) = Kg.z();

        Eigen::Matrix3d Ta_inv,Tg_inv;
        Eigen::Quaterniond Qai_inv;
        Ta_inv = Ta.inverse();
        Tg_inv = Tg.inverse();
        Quaterniond tmp_q = euler_to_quat(Qai);
        Qai_inv = tmp_q.inverse();
        */
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);
        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;
        //_jacobian = Get_imu_intr_J(_Ka, _Ma, _Kg, _Mg, _Qai);
        Eigen::Matrix3d dp_dKa = _jacobian.block<3, 3>(0, 0);
        Eigen::Matrix3d dp_dMa = _jacobian.block<3, 3>(0, 3);
        Eigen::Matrix3d dp_dKg = _jacobian.block<3, 3>(0, 6);
        Eigen::Matrix3d dp_dMg = _jacobian.block<3, 3>(0, 9);
        Eigen::Matrix3d dp_dQai= _jacobian.block<3, 3>(0, 12);

        Eigen::Matrix3d dq_dKg = _jacobian.block<3, 3>(3, 6);
        Eigen::Matrix3d dq_dMg = _jacobian.block<3, 3>(3, 9);

        Eigen::Matrix3d dv_dKa = _jacobian.block<3, 3>(6, 0);
        Eigen::Matrix3d dv_dMa = _jacobian.block<3, 3>(6, 3);
        Eigen::Matrix3d dv_dKg = _jacobian.block<3, 3>(6, 6);
        Eigen::Matrix3d dv_dMg = _jacobian.block<3, 3>(6, 9);
        Eigen::Matrix3d dv_dQai = _jacobian.block<3, 3>(6, 12);


        Eigen::Vector3d dKa = Ka - _Ka;
        Eigen::Vector3d dMa = Ma - _Ma;
        Eigen::Vector3d dKg = Kg - _Kg;
        Eigen::Vector3d dMg = Mg - _Mg;
        Eigen::Vector3d dQai = Qai - _Qai;



        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg)
                                             *Utility::deltaQ(dq_dKg * dKg) * Utility::deltaQ(dq_dMg * dMg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba *  dba + dv_dbg * dbg
                                          + dv_dKa * dKa + dv_dMa * dMa + dv_dKg* dKg + dv_dMg * dMg + dv_dQai * dQai;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba *  dba + dp_dbg * dbg
                                          + dp_dKa * dKa + dp_dMa * dMa + dp_dKg* dKg + dp_dMg * dMg + dp_dQai * dQai;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

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
    Eigen::Matrix<double, 9, 15>  _jacobian;
    Eigen::Matrix<double, 15, 15> _covariance;
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
