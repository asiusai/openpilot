#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_84199629467119576);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7936121580895356876);
void car_H_mod_fun(double *state, double *out_257442695438154731);
void car_f_fun(double *state, double dt, double *out_6052966079612224377);
void car_F_fun(double *state, double dt, double *out_7773349106694873210);
void car_h_25(double *state, double *unused, double *out_6566239321954013023);
void car_H_25(double *state, double *unused, double *out_509005066644052518);
void car_h_24(double *state, double *unused, double *out_2146863301753489992);
void car_H_24(double *state, double *unused, double *out_7338348944229969544);
void car_h_30(double *state, double *unused, double *out_8243055223501516243);
void car_H_30(double *state, double *unused, double *out_5036701396771660716);
void car_h_26(double *state, double *unused, double *out_7299545370599853808);
void car_H_26(double *state, double *unused, double *out_4250508385518108742);
void car_h_27(double *state, double *unused, double *out_3529453270509680338);
void car_H_27(double *state, double *unused, double *out_7211464708572085627);
void car_h_29(double *state, double *unused, double *out_3241381955840670598);
void car_H_29(double *state, double *unused, double *out_4526470052457268532);
void car_h_28(double *state, double *unused, double *out_7927405351791309350);
void car_H_28(double *state, double *unused, double *out_2562839780891942281);
void car_h_31(double *state, double *unused, double *out_625309790901498773);
void car_H_31(double *state, double *unused, double *out_4876716487751460218);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}