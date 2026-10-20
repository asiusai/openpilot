#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_2560052200344032530);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6680368481401804372);
void pose_H_mod_fun(double *state, double *out_3134495626797744797);
void pose_f_fun(double *state, double dt, double *out_3474868352977604468);
void pose_F_fun(double *state, double dt, double *out_7842334617962062267);
void pose_h_4(double *state, double *unused, double *out_8161452347818855726);
void pose_H_4(double *state, double *unused, double *out_7761974523197687256);
void pose_h_10(double *state, double *unused, double *out_5436618567726169234);
void pose_H_10(double *state, double *unused, double *out_7737771249214637356);
void pose_h_13(double *state, double *unused, double *out_4897505890852390831);
void pose_H_13(double *state, double *unused, double *out_7472495725179531559);
void pose_h_14(double *state, double *unused, double *out_5083935946789046305);
void pose_H_14(double *state, double *unused, double *out_6721528694172379831);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}