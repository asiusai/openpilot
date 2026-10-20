#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_84199629467119576) {
   out_84199629467119576[0] = delta_x[0] + nom_x[0];
   out_84199629467119576[1] = delta_x[1] + nom_x[1];
   out_84199629467119576[2] = delta_x[2] + nom_x[2];
   out_84199629467119576[3] = delta_x[3] + nom_x[3];
   out_84199629467119576[4] = delta_x[4] + nom_x[4];
   out_84199629467119576[5] = delta_x[5] + nom_x[5];
   out_84199629467119576[6] = delta_x[6] + nom_x[6];
   out_84199629467119576[7] = delta_x[7] + nom_x[7];
   out_84199629467119576[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7936121580895356876) {
   out_7936121580895356876[0] = -nom_x[0] + true_x[0];
   out_7936121580895356876[1] = -nom_x[1] + true_x[1];
   out_7936121580895356876[2] = -nom_x[2] + true_x[2];
   out_7936121580895356876[3] = -nom_x[3] + true_x[3];
   out_7936121580895356876[4] = -nom_x[4] + true_x[4];
   out_7936121580895356876[5] = -nom_x[5] + true_x[5];
   out_7936121580895356876[6] = -nom_x[6] + true_x[6];
   out_7936121580895356876[7] = -nom_x[7] + true_x[7];
   out_7936121580895356876[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_257442695438154731) {
   out_257442695438154731[0] = 1.0;
   out_257442695438154731[1] = 0.0;
   out_257442695438154731[2] = 0.0;
   out_257442695438154731[3] = 0.0;
   out_257442695438154731[4] = 0.0;
   out_257442695438154731[5] = 0.0;
   out_257442695438154731[6] = 0.0;
   out_257442695438154731[7] = 0.0;
   out_257442695438154731[8] = 0.0;
   out_257442695438154731[9] = 0.0;
   out_257442695438154731[10] = 1.0;
   out_257442695438154731[11] = 0.0;
   out_257442695438154731[12] = 0.0;
   out_257442695438154731[13] = 0.0;
   out_257442695438154731[14] = 0.0;
   out_257442695438154731[15] = 0.0;
   out_257442695438154731[16] = 0.0;
   out_257442695438154731[17] = 0.0;
   out_257442695438154731[18] = 0.0;
   out_257442695438154731[19] = 0.0;
   out_257442695438154731[20] = 1.0;
   out_257442695438154731[21] = 0.0;
   out_257442695438154731[22] = 0.0;
   out_257442695438154731[23] = 0.0;
   out_257442695438154731[24] = 0.0;
   out_257442695438154731[25] = 0.0;
   out_257442695438154731[26] = 0.0;
   out_257442695438154731[27] = 0.0;
   out_257442695438154731[28] = 0.0;
   out_257442695438154731[29] = 0.0;
   out_257442695438154731[30] = 1.0;
   out_257442695438154731[31] = 0.0;
   out_257442695438154731[32] = 0.0;
   out_257442695438154731[33] = 0.0;
   out_257442695438154731[34] = 0.0;
   out_257442695438154731[35] = 0.0;
   out_257442695438154731[36] = 0.0;
   out_257442695438154731[37] = 0.0;
   out_257442695438154731[38] = 0.0;
   out_257442695438154731[39] = 0.0;
   out_257442695438154731[40] = 1.0;
   out_257442695438154731[41] = 0.0;
   out_257442695438154731[42] = 0.0;
   out_257442695438154731[43] = 0.0;
   out_257442695438154731[44] = 0.0;
   out_257442695438154731[45] = 0.0;
   out_257442695438154731[46] = 0.0;
   out_257442695438154731[47] = 0.0;
   out_257442695438154731[48] = 0.0;
   out_257442695438154731[49] = 0.0;
   out_257442695438154731[50] = 1.0;
   out_257442695438154731[51] = 0.0;
   out_257442695438154731[52] = 0.0;
   out_257442695438154731[53] = 0.0;
   out_257442695438154731[54] = 0.0;
   out_257442695438154731[55] = 0.0;
   out_257442695438154731[56] = 0.0;
   out_257442695438154731[57] = 0.0;
   out_257442695438154731[58] = 0.0;
   out_257442695438154731[59] = 0.0;
   out_257442695438154731[60] = 1.0;
   out_257442695438154731[61] = 0.0;
   out_257442695438154731[62] = 0.0;
   out_257442695438154731[63] = 0.0;
   out_257442695438154731[64] = 0.0;
   out_257442695438154731[65] = 0.0;
   out_257442695438154731[66] = 0.0;
   out_257442695438154731[67] = 0.0;
   out_257442695438154731[68] = 0.0;
   out_257442695438154731[69] = 0.0;
   out_257442695438154731[70] = 1.0;
   out_257442695438154731[71] = 0.0;
   out_257442695438154731[72] = 0.0;
   out_257442695438154731[73] = 0.0;
   out_257442695438154731[74] = 0.0;
   out_257442695438154731[75] = 0.0;
   out_257442695438154731[76] = 0.0;
   out_257442695438154731[77] = 0.0;
   out_257442695438154731[78] = 0.0;
   out_257442695438154731[79] = 0.0;
   out_257442695438154731[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6052966079612224377) {
   out_6052966079612224377[0] = state[0];
   out_6052966079612224377[1] = state[1];
   out_6052966079612224377[2] = state[2];
   out_6052966079612224377[3] = state[3];
   out_6052966079612224377[4] = state[4];
   out_6052966079612224377[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_6052966079612224377[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6052966079612224377[7] = state[7];
   out_6052966079612224377[8] = state[8];
}
void F_fun(double *state, double dt, double *out_7773349106694873210) {
   out_7773349106694873210[0] = 1;
   out_7773349106694873210[1] = 0;
   out_7773349106694873210[2] = 0;
   out_7773349106694873210[3] = 0;
   out_7773349106694873210[4] = 0;
   out_7773349106694873210[5] = 0;
   out_7773349106694873210[6] = 0;
   out_7773349106694873210[7] = 0;
   out_7773349106694873210[8] = 0;
   out_7773349106694873210[9] = 0;
   out_7773349106694873210[10] = 1;
   out_7773349106694873210[11] = 0;
   out_7773349106694873210[12] = 0;
   out_7773349106694873210[13] = 0;
   out_7773349106694873210[14] = 0;
   out_7773349106694873210[15] = 0;
   out_7773349106694873210[16] = 0;
   out_7773349106694873210[17] = 0;
   out_7773349106694873210[18] = 0;
   out_7773349106694873210[19] = 0;
   out_7773349106694873210[20] = 1;
   out_7773349106694873210[21] = 0;
   out_7773349106694873210[22] = 0;
   out_7773349106694873210[23] = 0;
   out_7773349106694873210[24] = 0;
   out_7773349106694873210[25] = 0;
   out_7773349106694873210[26] = 0;
   out_7773349106694873210[27] = 0;
   out_7773349106694873210[28] = 0;
   out_7773349106694873210[29] = 0;
   out_7773349106694873210[30] = 1;
   out_7773349106694873210[31] = 0;
   out_7773349106694873210[32] = 0;
   out_7773349106694873210[33] = 0;
   out_7773349106694873210[34] = 0;
   out_7773349106694873210[35] = 0;
   out_7773349106694873210[36] = 0;
   out_7773349106694873210[37] = 0;
   out_7773349106694873210[38] = 0;
   out_7773349106694873210[39] = 0;
   out_7773349106694873210[40] = 1;
   out_7773349106694873210[41] = 0;
   out_7773349106694873210[42] = 0;
   out_7773349106694873210[43] = 0;
   out_7773349106694873210[44] = 0;
   out_7773349106694873210[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_7773349106694873210[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_7773349106694873210[47] = -dt*stiffness_front*state[0]/state[1];
   out_7773349106694873210[48] = -dt*stiffness_front*state[0]/state[1];
   out_7773349106694873210[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_7773349106694873210[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_7773349106694873210[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_7773349106694873210[52] = dt*stiffness_front*state[0]/state[1];
   out_7773349106694873210[53] = -9.8100000000000005*dt;
   out_7773349106694873210[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_7773349106694873210[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_7773349106694873210[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7773349106694873210[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7773349106694873210[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_7773349106694873210[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_7773349106694873210[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_7773349106694873210[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7773349106694873210[62] = 0;
   out_7773349106694873210[63] = 0;
   out_7773349106694873210[64] = 0;
   out_7773349106694873210[65] = 0;
   out_7773349106694873210[66] = 0;
   out_7773349106694873210[67] = 0;
   out_7773349106694873210[68] = 0;
   out_7773349106694873210[69] = 0;
   out_7773349106694873210[70] = 1;
   out_7773349106694873210[71] = 0;
   out_7773349106694873210[72] = 0;
   out_7773349106694873210[73] = 0;
   out_7773349106694873210[74] = 0;
   out_7773349106694873210[75] = 0;
   out_7773349106694873210[76] = 0;
   out_7773349106694873210[77] = 0;
   out_7773349106694873210[78] = 0;
   out_7773349106694873210[79] = 0;
   out_7773349106694873210[80] = 1;
}
void h_25(double *state, double *unused, double *out_6566239321954013023) {
   out_6566239321954013023[0] = state[6];
}
void H_25(double *state, double *unused, double *out_509005066644052518) {
   out_509005066644052518[0] = 0;
   out_509005066644052518[1] = 0;
   out_509005066644052518[2] = 0;
   out_509005066644052518[3] = 0;
   out_509005066644052518[4] = 0;
   out_509005066644052518[5] = 0;
   out_509005066644052518[6] = 1;
   out_509005066644052518[7] = 0;
   out_509005066644052518[8] = 0;
}
void h_24(double *state, double *unused, double *out_2146863301753489992) {
   out_2146863301753489992[0] = state[4];
   out_2146863301753489992[1] = state[5];
}
void H_24(double *state, double *unused, double *out_7338348944229969544) {
   out_7338348944229969544[0] = 0;
   out_7338348944229969544[1] = 0;
   out_7338348944229969544[2] = 0;
   out_7338348944229969544[3] = 0;
   out_7338348944229969544[4] = 1;
   out_7338348944229969544[5] = 0;
   out_7338348944229969544[6] = 0;
   out_7338348944229969544[7] = 0;
   out_7338348944229969544[8] = 0;
   out_7338348944229969544[9] = 0;
   out_7338348944229969544[10] = 0;
   out_7338348944229969544[11] = 0;
   out_7338348944229969544[12] = 0;
   out_7338348944229969544[13] = 0;
   out_7338348944229969544[14] = 1;
   out_7338348944229969544[15] = 0;
   out_7338348944229969544[16] = 0;
   out_7338348944229969544[17] = 0;
}
void h_30(double *state, double *unused, double *out_8243055223501516243) {
   out_8243055223501516243[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5036701396771660716) {
   out_5036701396771660716[0] = 0;
   out_5036701396771660716[1] = 0;
   out_5036701396771660716[2] = 0;
   out_5036701396771660716[3] = 0;
   out_5036701396771660716[4] = 1;
   out_5036701396771660716[5] = 0;
   out_5036701396771660716[6] = 0;
   out_5036701396771660716[7] = 0;
   out_5036701396771660716[8] = 0;
}
void h_26(double *state, double *unused, double *out_7299545370599853808) {
   out_7299545370599853808[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4250508385518108742) {
   out_4250508385518108742[0] = 0;
   out_4250508385518108742[1] = 0;
   out_4250508385518108742[2] = 0;
   out_4250508385518108742[3] = 0;
   out_4250508385518108742[4] = 0;
   out_4250508385518108742[5] = 0;
   out_4250508385518108742[6] = 0;
   out_4250508385518108742[7] = 1;
   out_4250508385518108742[8] = 0;
}
void h_27(double *state, double *unused, double *out_3529453270509680338) {
   out_3529453270509680338[0] = state[3];
}
void H_27(double *state, double *unused, double *out_7211464708572085627) {
   out_7211464708572085627[0] = 0;
   out_7211464708572085627[1] = 0;
   out_7211464708572085627[2] = 0;
   out_7211464708572085627[3] = 1;
   out_7211464708572085627[4] = 0;
   out_7211464708572085627[5] = 0;
   out_7211464708572085627[6] = 0;
   out_7211464708572085627[7] = 0;
   out_7211464708572085627[8] = 0;
}
void h_29(double *state, double *unused, double *out_3241381955840670598) {
   out_3241381955840670598[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4526470052457268532) {
   out_4526470052457268532[0] = 0;
   out_4526470052457268532[1] = 1;
   out_4526470052457268532[2] = 0;
   out_4526470052457268532[3] = 0;
   out_4526470052457268532[4] = 0;
   out_4526470052457268532[5] = 0;
   out_4526470052457268532[6] = 0;
   out_4526470052457268532[7] = 0;
   out_4526470052457268532[8] = 0;
}
void h_28(double *state, double *unused, double *out_7927405351791309350) {
   out_7927405351791309350[0] = state[0];
}
void H_28(double *state, double *unused, double *out_2562839780891942281) {
   out_2562839780891942281[0] = 1;
   out_2562839780891942281[1] = 0;
   out_2562839780891942281[2] = 0;
   out_2562839780891942281[3] = 0;
   out_2562839780891942281[4] = 0;
   out_2562839780891942281[5] = 0;
   out_2562839780891942281[6] = 0;
   out_2562839780891942281[7] = 0;
   out_2562839780891942281[8] = 0;
}
void h_31(double *state, double *unused, double *out_625309790901498773) {
   out_625309790901498773[0] = state[8];
}
void H_31(double *state, double *unused, double *out_4876716487751460218) {
   out_4876716487751460218[0] = 0;
   out_4876716487751460218[1] = 0;
   out_4876716487751460218[2] = 0;
   out_4876716487751460218[3] = 0;
   out_4876716487751460218[4] = 0;
   out_4876716487751460218[5] = 0;
   out_4876716487751460218[6] = 0;
   out_4876716487751460218[7] = 0;
   out_4876716487751460218[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_84199629467119576) {
  err_fun(nom_x, delta_x, out_84199629467119576);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7936121580895356876) {
  inv_err_fun(nom_x, true_x, out_7936121580895356876);
}
void car_H_mod_fun(double *state, double *out_257442695438154731) {
  H_mod_fun(state, out_257442695438154731);
}
void car_f_fun(double *state, double dt, double *out_6052966079612224377) {
  f_fun(state,  dt, out_6052966079612224377);
}
void car_F_fun(double *state, double dt, double *out_7773349106694873210) {
  F_fun(state,  dt, out_7773349106694873210);
}
void car_h_25(double *state, double *unused, double *out_6566239321954013023) {
  h_25(state, unused, out_6566239321954013023);
}
void car_H_25(double *state, double *unused, double *out_509005066644052518) {
  H_25(state, unused, out_509005066644052518);
}
void car_h_24(double *state, double *unused, double *out_2146863301753489992) {
  h_24(state, unused, out_2146863301753489992);
}
void car_H_24(double *state, double *unused, double *out_7338348944229969544) {
  H_24(state, unused, out_7338348944229969544);
}
void car_h_30(double *state, double *unused, double *out_8243055223501516243) {
  h_30(state, unused, out_8243055223501516243);
}
void car_H_30(double *state, double *unused, double *out_5036701396771660716) {
  H_30(state, unused, out_5036701396771660716);
}
void car_h_26(double *state, double *unused, double *out_7299545370599853808) {
  h_26(state, unused, out_7299545370599853808);
}
void car_H_26(double *state, double *unused, double *out_4250508385518108742) {
  H_26(state, unused, out_4250508385518108742);
}
void car_h_27(double *state, double *unused, double *out_3529453270509680338) {
  h_27(state, unused, out_3529453270509680338);
}
void car_H_27(double *state, double *unused, double *out_7211464708572085627) {
  H_27(state, unused, out_7211464708572085627);
}
void car_h_29(double *state, double *unused, double *out_3241381955840670598) {
  h_29(state, unused, out_3241381955840670598);
}
void car_H_29(double *state, double *unused, double *out_4526470052457268532) {
  H_29(state, unused, out_4526470052457268532);
}
void car_h_28(double *state, double *unused, double *out_7927405351791309350) {
  h_28(state, unused, out_7927405351791309350);
}
void car_H_28(double *state, double *unused, double *out_2562839780891942281) {
  H_28(state, unused, out_2562839780891942281);
}
void car_h_31(double *state, double *unused, double *out_625309790901498773) {
  h_31(state, unused, out_625309790901498773);
}
void car_H_31(double *state, double *unused, double *out_4876716487751460218) {
  H_31(state, unused, out_4876716487751460218);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
