#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_2560052200344032530) {
   out_2560052200344032530[0] = delta_x[0] + nom_x[0];
   out_2560052200344032530[1] = delta_x[1] + nom_x[1];
   out_2560052200344032530[2] = delta_x[2] + nom_x[2];
   out_2560052200344032530[3] = delta_x[3] + nom_x[3];
   out_2560052200344032530[4] = delta_x[4] + nom_x[4];
   out_2560052200344032530[5] = delta_x[5] + nom_x[5];
   out_2560052200344032530[6] = delta_x[6] + nom_x[6];
   out_2560052200344032530[7] = delta_x[7] + nom_x[7];
   out_2560052200344032530[8] = delta_x[8] + nom_x[8];
   out_2560052200344032530[9] = delta_x[9] + nom_x[9];
   out_2560052200344032530[10] = delta_x[10] + nom_x[10];
   out_2560052200344032530[11] = delta_x[11] + nom_x[11];
   out_2560052200344032530[12] = delta_x[12] + nom_x[12];
   out_2560052200344032530[13] = delta_x[13] + nom_x[13];
   out_2560052200344032530[14] = delta_x[14] + nom_x[14];
   out_2560052200344032530[15] = delta_x[15] + nom_x[15];
   out_2560052200344032530[16] = delta_x[16] + nom_x[16];
   out_2560052200344032530[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6680368481401804372) {
   out_6680368481401804372[0] = -nom_x[0] + true_x[0];
   out_6680368481401804372[1] = -nom_x[1] + true_x[1];
   out_6680368481401804372[2] = -nom_x[2] + true_x[2];
   out_6680368481401804372[3] = -nom_x[3] + true_x[3];
   out_6680368481401804372[4] = -nom_x[4] + true_x[4];
   out_6680368481401804372[5] = -nom_x[5] + true_x[5];
   out_6680368481401804372[6] = -nom_x[6] + true_x[6];
   out_6680368481401804372[7] = -nom_x[7] + true_x[7];
   out_6680368481401804372[8] = -nom_x[8] + true_x[8];
   out_6680368481401804372[9] = -nom_x[9] + true_x[9];
   out_6680368481401804372[10] = -nom_x[10] + true_x[10];
   out_6680368481401804372[11] = -nom_x[11] + true_x[11];
   out_6680368481401804372[12] = -nom_x[12] + true_x[12];
   out_6680368481401804372[13] = -nom_x[13] + true_x[13];
   out_6680368481401804372[14] = -nom_x[14] + true_x[14];
   out_6680368481401804372[15] = -nom_x[15] + true_x[15];
   out_6680368481401804372[16] = -nom_x[16] + true_x[16];
   out_6680368481401804372[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3134495626797744797) {
   out_3134495626797744797[0] = 1.0;
   out_3134495626797744797[1] = 0.0;
   out_3134495626797744797[2] = 0.0;
   out_3134495626797744797[3] = 0.0;
   out_3134495626797744797[4] = 0.0;
   out_3134495626797744797[5] = 0.0;
   out_3134495626797744797[6] = 0.0;
   out_3134495626797744797[7] = 0.0;
   out_3134495626797744797[8] = 0.0;
   out_3134495626797744797[9] = 0.0;
   out_3134495626797744797[10] = 0.0;
   out_3134495626797744797[11] = 0.0;
   out_3134495626797744797[12] = 0.0;
   out_3134495626797744797[13] = 0.0;
   out_3134495626797744797[14] = 0.0;
   out_3134495626797744797[15] = 0.0;
   out_3134495626797744797[16] = 0.0;
   out_3134495626797744797[17] = 0.0;
   out_3134495626797744797[18] = 0.0;
   out_3134495626797744797[19] = 1.0;
   out_3134495626797744797[20] = 0.0;
   out_3134495626797744797[21] = 0.0;
   out_3134495626797744797[22] = 0.0;
   out_3134495626797744797[23] = 0.0;
   out_3134495626797744797[24] = 0.0;
   out_3134495626797744797[25] = 0.0;
   out_3134495626797744797[26] = 0.0;
   out_3134495626797744797[27] = 0.0;
   out_3134495626797744797[28] = 0.0;
   out_3134495626797744797[29] = 0.0;
   out_3134495626797744797[30] = 0.0;
   out_3134495626797744797[31] = 0.0;
   out_3134495626797744797[32] = 0.0;
   out_3134495626797744797[33] = 0.0;
   out_3134495626797744797[34] = 0.0;
   out_3134495626797744797[35] = 0.0;
   out_3134495626797744797[36] = 0.0;
   out_3134495626797744797[37] = 0.0;
   out_3134495626797744797[38] = 1.0;
   out_3134495626797744797[39] = 0.0;
   out_3134495626797744797[40] = 0.0;
   out_3134495626797744797[41] = 0.0;
   out_3134495626797744797[42] = 0.0;
   out_3134495626797744797[43] = 0.0;
   out_3134495626797744797[44] = 0.0;
   out_3134495626797744797[45] = 0.0;
   out_3134495626797744797[46] = 0.0;
   out_3134495626797744797[47] = 0.0;
   out_3134495626797744797[48] = 0.0;
   out_3134495626797744797[49] = 0.0;
   out_3134495626797744797[50] = 0.0;
   out_3134495626797744797[51] = 0.0;
   out_3134495626797744797[52] = 0.0;
   out_3134495626797744797[53] = 0.0;
   out_3134495626797744797[54] = 0.0;
   out_3134495626797744797[55] = 0.0;
   out_3134495626797744797[56] = 0.0;
   out_3134495626797744797[57] = 1.0;
   out_3134495626797744797[58] = 0.0;
   out_3134495626797744797[59] = 0.0;
   out_3134495626797744797[60] = 0.0;
   out_3134495626797744797[61] = 0.0;
   out_3134495626797744797[62] = 0.0;
   out_3134495626797744797[63] = 0.0;
   out_3134495626797744797[64] = 0.0;
   out_3134495626797744797[65] = 0.0;
   out_3134495626797744797[66] = 0.0;
   out_3134495626797744797[67] = 0.0;
   out_3134495626797744797[68] = 0.0;
   out_3134495626797744797[69] = 0.0;
   out_3134495626797744797[70] = 0.0;
   out_3134495626797744797[71] = 0.0;
   out_3134495626797744797[72] = 0.0;
   out_3134495626797744797[73] = 0.0;
   out_3134495626797744797[74] = 0.0;
   out_3134495626797744797[75] = 0.0;
   out_3134495626797744797[76] = 1.0;
   out_3134495626797744797[77] = 0.0;
   out_3134495626797744797[78] = 0.0;
   out_3134495626797744797[79] = 0.0;
   out_3134495626797744797[80] = 0.0;
   out_3134495626797744797[81] = 0.0;
   out_3134495626797744797[82] = 0.0;
   out_3134495626797744797[83] = 0.0;
   out_3134495626797744797[84] = 0.0;
   out_3134495626797744797[85] = 0.0;
   out_3134495626797744797[86] = 0.0;
   out_3134495626797744797[87] = 0.0;
   out_3134495626797744797[88] = 0.0;
   out_3134495626797744797[89] = 0.0;
   out_3134495626797744797[90] = 0.0;
   out_3134495626797744797[91] = 0.0;
   out_3134495626797744797[92] = 0.0;
   out_3134495626797744797[93] = 0.0;
   out_3134495626797744797[94] = 0.0;
   out_3134495626797744797[95] = 1.0;
   out_3134495626797744797[96] = 0.0;
   out_3134495626797744797[97] = 0.0;
   out_3134495626797744797[98] = 0.0;
   out_3134495626797744797[99] = 0.0;
   out_3134495626797744797[100] = 0.0;
   out_3134495626797744797[101] = 0.0;
   out_3134495626797744797[102] = 0.0;
   out_3134495626797744797[103] = 0.0;
   out_3134495626797744797[104] = 0.0;
   out_3134495626797744797[105] = 0.0;
   out_3134495626797744797[106] = 0.0;
   out_3134495626797744797[107] = 0.0;
   out_3134495626797744797[108] = 0.0;
   out_3134495626797744797[109] = 0.0;
   out_3134495626797744797[110] = 0.0;
   out_3134495626797744797[111] = 0.0;
   out_3134495626797744797[112] = 0.0;
   out_3134495626797744797[113] = 0.0;
   out_3134495626797744797[114] = 1.0;
   out_3134495626797744797[115] = 0.0;
   out_3134495626797744797[116] = 0.0;
   out_3134495626797744797[117] = 0.0;
   out_3134495626797744797[118] = 0.0;
   out_3134495626797744797[119] = 0.0;
   out_3134495626797744797[120] = 0.0;
   out_3134495626797744797[121] = 0.0;
   out_3134495626797744797[122] = 0.0;
   out_3134495626797744797[123] = 0.0;
   out_3134495626797744797[124] = 0.0;
   out_3134495626797744797[125] = 0.0;
   out_3134495626797744797[126] = 0.0;
   out_3134495626797744797[127] = 0.0;
   out_3134495626797744797[128] = 0.0;
   out_3134495626797744797[129] = 0.0;
   out_3134495626797744797[130] = 0.0;
   out_3134495626797744797[131] = 0.0;
   out_3134495626797744797[132] = 0.0;
   out_3134495626797744797[133] = 1.0;
   out_3134495626797744797[134] = 0.0;
   out_3134495626797744797[135] = 0.0;
   out_3134495626797744797[136] = 0.0;
   out_3134495626797744797[137] = 0.0;
   out_3134495626797744797[138] = 0.0;
   out_3134495626797744797[139] = 0.0;
   out_3134495626797744797[140] = 0.0;
   out_3134495626797744797[141] = 0.0;
   out_3134495626797744797[142] = 0.0;
   out_3134495626797744797[143] = 0.0;
   out_3134495626797744797[144] = 0.0;
   out_3134495626797744797[145] = 0.0;
   out_3134495626797744797[146] = 0.0;
   out_3134495626797744797[147] = 0.0;
   out_3134495626797744797[148] = 0.0;
   out_3134495626797744797[149] = 0.0;
   out_3134495626797744797[150] = 0.0;
   out_3134495626797744797[151] = 0.0;
   out_3134495626797744797[152] = 1.0;
   out_3134495626797744797[153] = 0.0;
   out_3134495626797744797[154] = 0.0;
   out_3134495626797744797[155] = 0.0;
   out_3134495626797744797[156] = 0.0;
   out_3134495626797744797[157] = 0.0;
   out_3134495626797744797[158] = 0.0;
   out_3134495626797744797[159] = 0.0;
   out_3134495626797744797[160] = 0.0;
   out_3134495626797744797[161] = 0.0;
   out_3134495626797744797[162] = 0.0;
   out_3134495626797744797[163] = 0.0;
   out_3134495626797744797[164] = 0.0;
   out_3134495626797744797[165] = 0.0;
   out_3134495626797744797[166] = 0.0;
   out_3134495626797744797[167] = 0.0;
   out_3134495626797744797[168] = 0.0;
   out_3134495626797744797[169] = 0.0;
   out_3134495626797744797[170] = 0.0;
   out_3134495626797744797[171] = 1.0;
   out_3134495626797744797[172] = 0.0;
   out_3134495626797744797[173] = 0.0;
   out_3134495626797744797[174] = 0.0;
   out_3134495626797744797[175] = 0.0;
   out_3134495626797744797[176] = 0.0;
   out_3134495626797744797[177] = 0.0;
   out_3134495626797744797[178] = 0.0;
   out_3134495626797744797[179] = 0.0;
   out_3134495626797744797[180] = 0.0;
   out_3134495626797744797[181] = 0.0;
   out_3134495626797744797[182] = 0.0;
   out_3134495626797744797[183] = 0.0;
   out_3134495626797744797[184] = 0.0;
   out_3134495626797744797[185] = 0.0;
   out_3134495626797744797[186] = 0.0;
   out_3134495626797744797[187] = 0.0;
   out_3134495626797744797[188] = 0.0;
   out_3134495626797744797[189] = 0.0;
   out_3134495626797744797[190] = 1.0;
   out_3134495626797744797[191] = 0.0;
   out_3134495626797744797[192] = 0.0;
   out_3134495626797744797[193] = 0.0;
   out_3134495626797744797[194] = 0.0;
   out_3134495626797744797[195] = 0.0;
   out_3134495626797744797[196] = 0.0;
   out_3134495626797744797[197] = 0.0;
   out_3134495626797744797[198] = 0.0;
   out_3134495626797744797[199] = 0.0;
   out_3134495626797744797[200] = 0.0;
   out_3134495626797744797[201] = 0.0;
   out_3134495626797744797[202] = 0.0;
   out_3134495626797744797[203] = 0.0;
   out_3134495626797744797[204] = 0.0;
   out_3134495626797744797[205] = 0.0;
   out_3134495626797744797[206] = 0.0;
   out_3134495626797744797[207] = 0.0;
   out_3134495626797744797[208] = 0.0;
   out_3134495626797744797[209] = 1.0;
   out_3134495626797744797[210] = 0.0;
   out_3134495626797744797[211] = 0.0;
   out_3134495626797744797[212] = 0.0;
   out_3134495626797744797[213] = 0.0;
   out_3134495626797744797[214] = 0.0;
   out_3134495626797744797[215] = 0.0;
   out_3134495626797744797[216] = 0.0;
   out_3134495626797744797[217] = 0.0;
   out_3134495626797744797[218] = 0.0;
   out_3134495626797744797[219] = 0.0;
   out_3134495626797744797[220] = 0.0;
   out_3134495626797744797[221] = 0.0;
   out_3134495626797744797[222] = 0.0;
   out_3134495626797744797[223] = 0.0;
   out_3134495626797744797[224] = 0.0;
   out_3134495626797744797[225] = 0.0;
   out_3134495626797744797[226] = 0.0;
   out_3134495626797744797[227] = 0.0;
   out_3134495626797744797[228] = 1.0;
   out_3134495626797744797[229] = 0.0;
   out_3134495626797744797[230] = 0.0;
   out_3134495626797744797[231] = 0.0;
   out_3134495626797744797[232] = 0.0;
   out_3134495626797744797[233] = 0.0;
   out_3134495626797744797[234] = 0.0;
   out_3134495626797744797[235] = 0.0;
   out_3134495626797744797[236] = 0.0;
   out_3134495626797744797[237] = 0.0;
   out_3134495626797744797[238] = 0.0;
   out_3134495626797744797[239] = 0.0;
   out_3134495626797744797[240] = 0.0;
   out_3134495626797744797[241] = 0.0;
   out_3134495626797744797[242] = 0.0;
   out_3134495626797744797[243] = 0.0;
   out_3134495626797744797[244] = 0.0;
   out_3134495626797744797[245] = 0.0;
   out_3134495626797744797[246] = 0.0;
   out_3134495626797744797[247] = 1.0;
   out_3134495626797744797[248] = 0.0;
   out_3134495626797744797[249] = 0.0;
   out_3134495626797744797[250] = 0.0;
   out_3134495626797744797[251] = 0.0;
   out_3134495626797744797[252] = 0.0;
   out_3134495626797744797[253] = 0.0;
   out_3134495626797744797[254] = 0.0;
   out_3134495626797744797[255] = 0.0;
   out_3134495626797744797[256] = 0.0;
   out_3134495626797744797[257] = 0.0;
   out_3134495626797744797[258] = 0.0;
   out_3134495626797744797[259] = 0.0;
   out_3134495626797744797[260] = 0.0;
   out_3134495626797744797[261] = 0.0;
   out_3134495626797744797[262] = 0.0;
   out_3134495626797744797[263] = 0.0;
   out_3134495626797744797[264] = 0.0;
   out_3134495626797744797[265] = 0.0;
   out_3134495626797744797[266] = 1.0;
   out_3134495626797744797[267] = 0.0;
   out_3134495626797744797[268] = 0.0;
   out_3134495626797744797[269] = 0.0;
   out_3134495626797744797[270] = 0.0;
   out_3134495626797744797[271] = 0.0;
   out_3134495626797744797[272] = 0.0;
   out_3134495626797744797[273] = 0.0;
   out_3134495626797744797[274] = 0.0;
   out_3134495626797744797[275] = 0.0;
   out_3134495626797744797[276] = 0.0;
   out_3134495626797744797[277] = 0.0;
   out_3134495626797744797[278] = 0.0;
   out_3134495626797744797[279] = 0.0;
   out_3134495626797744797[280] = 0.0;
   out_3134495626797744797[281] = 0.0;
   out_3134495626797744797[282] = 0.0;
   out_3134495626797744797[283] = 0.0;
   out_3134495626797744797[284] = 0.0;
   out_3134495626797744797[285] = 1.0;
   out_3134495626797744797[286] = 0.0;
   out_3134495626797744797[287] = 0.0;
   out_3134495626797744797[288] = 0.0;
   out_3134495626797744797[289] = 0.0;
   out_3134495626797744797[290] = 0.0;
   out_3134495626797744797[291] = 0.0;
   out_3134495626797744797[292] = 0.0;
   out_3134495626797744797[293] = 0.0;
   out_3134495626797744797[294] = 0.0;
   out_3134495626797744797[295] = 0.0;
   out_3134495626797744797[296] = 0.0;
   out_3134495626797744797[297] = 0.0;
   out_3134495626797744797[298] = 0.0;
   out_3134495626797744797[299] = 0.0;
   out_3134495626797744797[300] = 0.0;
   out_3134495626797744797[301] = 0.0;
   out_3134495626797744797[302] = 0.0;
   out_3134495626797744797[303] = 0.0;
   out_3134495626797744797[304] = 1.0;
   out_3134495626797744797[305] = 0.0;
   out_3134495626797744797[306] = 0.0;
   out_3134495626797744797[307] = 0.0;
   out_3134495626797744797[308] = 0.0;
   out_3134495626797744797[309] = 0.0;
   out_3134495626797744797[310] = 0.0;
   out_3134495626797744797[311] = 0.0;
   out_3134495626797744797[312] = 0.0;
   out_3134495626797744797[313] = 0.0;
   out_3134495626797744797[314] = 0.0;
   out_3134495626797744797[315] = 0.0;
   out_3134495626797744797[316] = 0.0;
   out_3134495626797744797[317] = 0.0;
   out_3134495626797744797[318] = 0.0;
   out_3134495626797744797[319] = 0.0;
   out_3134495626797744797[320] = 0.0;
   out_3134495626797744797[321] = 0.0;
   out_3134495626797744797[322] = 0.0;
   out_3134495626797744797[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_3474868352977604468) {
   out_3474868352977604468[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_3474868352977604468[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_3474868352977604468[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_3474868352977604468[3] = dt*state[12] + state[3];
   out_3474868352977604468[4] = dt*state[13] + state[4];
   out_3474868352977604468[5] = dt*state[14] + state[5];
   out_3474868352977604468[6] = state[6];
   out_3474868352977604468[7] = state[7];
   out_3474868352977604468[8] = state[8];
   out_3474868352977604468[9] = state[9];
   out_3474868352977604468[10] = state[10];
   out_3474868352977604468[11] = state[11];
   out_3474868352977604468[12] = state[12];
   out_3474868352977604468[13] = state[13];
   out_3474868352977604468[14] = state[14];
   out_3474868352977604468[15] = state[15];
   out_3474868352977604468[16] = state[16];
   out_3474868352977604468[17] = state[17];
}
void F_fun(double *state, double dt, double *out_7842334617962062267) {
   out_7842334617962062267[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7842334617962062267[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7842334617962062267[2] = 0;
   out_7842334617962062267[3] = 0;
   out_7842334617962062267[4] = 0;
   out_7842334617962062267[5] = 0;
   out_7842334617962062267[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7842334617962062267[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7842334617962062267[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7842334617962062267[9] = 0;
   out_7842334617962062267[10] = 0;
   out_7842334617962062267[11] = 0;
   out_7842334617962062267[12] = 0;
   out_7842334617962062267[13] = 0;
   out_7842334617962062267[14] = 0;
   out_7842334617962062267[15] = 0;
   out_7842334617962062267[16] = 0;
   out_7842334617962062267[17] = 0;
   out_7842334617962062267[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7842334617962062267[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7842334617962062267[20] = 0;
   out_7842334617962062267[21] = 0;
   out_7842334617962062267[22] = 0;
   out_7842334617962062267[23] = 0;
   out_7842334617962062267[24] = 0;
   out_7842334617962062267[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7842334617962062267[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7842334617962062267[27] = 0;
   out_7842334617962062267[28] = 0;
   out_7842334617962062267[29] = 0;
   out_7842334617962062267[30] = 0;
   out_7842334617962062267[31] = 0;
   out_7842334617962062267[32] = 0;
   out_7842334617962062267[33] = 0;
   out_7842334617962062267[34] = 0;
   out_7842334617962062267[35] = 0;
   out_7842334617962062267[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7842334617962062267[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7842334617962062267[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7842334617962062267[39] = 0;
   out_7842334617962062267[40] = 0;
   out_7842334617962062267[41] = 0;
   out_7842334617962062267[42] = 0;
   out_7842334617962062267[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7842334617962062267[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7842334617962062267[45] = 0;
   out_7842334617962062267[46] = 0;
   out_7842334617962062267[47] = 0;
   out_7842334617962062267[48] = 0;
   out_7842334617962062267[49] = 0;
   out_7842334617962062267[50] = 0;
   out_7842334617962062267[51] = 0;
   out_7842334617962062267[52] = 0;
   out_7842334617962062267[53] = 0;
   out_7842334617962062267[54] = 0;
   out_7842334617962062267[55] = 0;
   out_7842334617962062267[56] = 0;
   out_7842334617962062267[57] = 1;
   out_7842334617962062267[58] = 0;
   out_7842334617962062267[59] = 0;
   out_7842334617962062267[60] = 0;
   out_7842334617962062267[61] = 0;
   out_7842334617962062267[62] = 0;
   out_7842334617962062267[63] = 0;
   out_7842334617962062267[64] = 0;
   out_7842334617962062267[65] = 0;
   out_7842334617962062267[66] = dt;
   out_7842334617962062267[67] = 0;
   out_7842334617962062267[68] = 0;
   out_7842334617962062267[69] = 0;
   out_7842334617962062267[70] = 0;
   out_7842334617962062267[71] = 0;
   out_7842334617962062267[72] = 0;
   out_7842334617962062267[73] = 0;
   out_7842334617962062267[74] = 0;
   out_7842334617962062267[75] = 0;
   out_7842334617962062267[76] = 1;
   out_7842334617962062267[77] = 0;
   out_7842334617962062267[78] = 0;
   out_7842334617962062267[79] = 0;
   out_7842334617962062267[80] = 0;
   out_7842334617962062267[81] = 0;
   out_7842334617962062267[82] = 0;
   out_7842334617962062267[83] = 0;
   out_7842334617962062267[84] = 0;
   out_7842334617962062267[85] = dt;
   out_7842334617962062267[86] = 0;
   out_7842334617962062267[87] = 0;
   out_7842334617962062267[88] = 0;
   out_7842334617962062267[89] = 0;
   out_7842334617962062267[90] = 0;
   out_7842334617962062267[91] = 0;
   out_7842334617962062267[92] = 0;
   out_7842334617962062267[93] = 0;
   out_7842334617962062267[94] = 0;
   out_7842334617962062267[95] = 1;
   out_7842334617962062267[96] = 0;
   out_7842334617962062267[97] = 0;
   out_7842334617962062267[98] = 0;
   out_7842334617962062267[99] = 0;
   out_7842334617962062267[100] = 0;
   out_7842334617962062267[101] = 0;
   out_7842334617962062267[102] = 0;
   out_7842334617962062267[103] = 0;
   out_7842334617962062267[104] = dt;
   out_7842334617962062267[105] = 0;
   out_7842334617962062267[106] = 0;
   out_7842334617962062267[107] = 0;
   out_7842334617962062267[108] = 0;
   out_7842334617962062267[109] = 0;
   out_7842334617962062267[110] = 0;
   out_7842334617962062267[111] = 0;
   out_7842334617962062267[112] = 0;
   out_7842334617962062267[113] = 0;
   out_7842334617962062267[114] = 1;
   out_7842334617962062267[115] = 0;
   out_7842334617962062267[116] = 0;
   out_7842334617962062267[117] = 0;
   out_7842334617962062267[118] = 0;
   out_7842334617962062267[119] = 0;
   out_7842334617962062267[120] = 0;
   out_7842334617962062267[121] = 0;
   out_7842334617962062267[122] = 0;
   out_7842334617962062267[123] = 0;
   out_7842334617962062267[124] = 0;
   out_7842334617962062267[125] = 0;
   out_7842334617962062267[126] = 0;
   out_7842334617962062267[127] = 0;
   out_7842334617962062267[128] = 0;
   out_7842334617962062267[129] = 0;
   out_7842334617962062267[130] = 0;
   out_7842334617962062267[131] = 0;
   out_7842334617962062267[132] = 0;
   out_7842334617962062267[133] = 1;
   out_7842334617962062267[134] = 0;
   out_7842334617962062267[135] = 0;
   out_7842334617962062267[136] = 0;
   out_7842334617962062267[137] = 0;
   out_7842334617962062267[138] = 0;
   out_7842334617962062267[139] = 0;
   out_7842334617962062267[140] = 0;
   out_7842334617962062267[141] = 0;
   out_7842334617962062267[142] = 0;
   out_7842334617962062267[143] = 0;
   out_7842334617962062267[144] = 0;
   out_7842334617962062267[145] = 0;
   out_7842334617962062267[146] = 0;
   out_7842334617962062267[147] = 0;
   out_7842334617962062267[148] = 0;
   out_7842334617962062267[149] = 0;
   out_7842334617962062267[150] = 0;
   out_7842334617962062267[151] = 0;
   out_7842334617962062267[152] = 1;
   out_7842334617962062267[153] = 0;
   out_7842334617962062267[154] = 0;
   out_7842334617962062267[155] = 0;
   out_7842334617962062267[156] = 0;
   out_7842334617962062267[157] = 0;
   out_7842334617962062267[158] = 0;
   out_7842334617962062267[159] = 0;
   out_7842334617962062267[160] = 0;
   out_7842334617962062267[161] = 0;
   out_7842334617962062267[162] = 0;
   out_7842334617962062267[163] = 0;
   out_7842334617962062267[164] = 0;
   out_7842334617962062267[165] = 0;
   out_7842334617962062267[166] = 0;
   out_7842334617962062267[167] = 0;
   out_7842334617962062267[168] = 0;
   out_7842334617962062267[169] = 0;
   out_7842334617962062267[170] = 0;
   out_7842334617962062267[171] = 1;
   out_7842334617962062267[172] = 0;
   out_7842334617962062267[173] = 0;
   out_7842334617962062267[174] = 0;
   out_7842334617962062267[175] = 0;
   out_7842334617962062267[176] = 0;
   out_7842334617962062267[177] = 0;
   out_7842334617962062267[178] = 0;
   out_7842334617962062267[179] = 0;
   out_7842334617962062267[180] = 0;
   out_7842334617962062267[181] = 0;
   out_7842334617962062267[182] = 0;
   out_7842334617962062267[183] = 0;
   out_7842334617962062267[184] = 0;
   out_7842334617962062267[185] = 0;
   out_7842334617962062267[186] = 0;
   out_7842334617962062267[187] = 0;
   out_7842334617962062267[188] = 0;
   out_7842334617962062267[189] = 0;
   out_7842334617962062267[190] = 1;
   out_7842334617962062267[191] = 0;
   out_7842334617962062267[192] = 0;
   out_7842334617962062267[193] = 0;
   out_7842334617962062267[194] = 0;
   out_7842334617962062267[195] = 0;
   out_7842334617962062267[196] = 0;
   out_7842334617962062267[197] = 0;
   out_7842334617962062267[198] = 0;
   out_7842334617962062267[199] = 0;
   out_7842334617962062267[200] = 0;
   out_7842334617962062267[201] = 0;
   out_7842334617962062267[202] = 0;
   out_7842334617962062267[203] = 0;
   out_7842334617962062267[204] = 0;
   out_7842334617962062267[205] = 0;
   out_7842334617962062267[206] = 0;
   out_7842334617962062267[207] = 0;
   out_7842334617962062267[208] = 0;
   out_7842334617962062267[209] = 1;
   out_7842334617962062267[210] = 0;
   out_7842334617962062267[211] = 0;
   out_7842334617962062267[212] = 0;
   out_7842334617962062267[213] = 0;
   out_7842334617962062267[214] = 0;
   out_7842334617962062267[215] = 0;
   out_7842334617962062267[216] = 0;
   out_7842334617962062267[217] = 0;
   out_7842334617962062267[218] = 0;
   out_7842334617962062267[219] = 0;
   out_7842334617962062267[220] = 0;
   out_7842334617962062267[221] = 0;
   out_7842334617962062267[222] = 0;
   out_7842334617962062267[223] = 0;
   out_7842334617962062267[224] = 0;
   out_7842334617962062267[225] = 0;
   out_7842334617962062267[226] = 0;
   out_7842334617962062267[227] = 0;
   out_7842334617962062267[228] = 1;
   out_7842334617962062267[229] = 0;
   out_7842334617962062267[230] = 0;
   out_7842334617962062267[231] = 0;
   out_7842334617962062267[232] = 0;
   out_7842334617962062267[233] = 0;
   out_7842334617962062267[234] = 0;
   out_7842334617962062267[235] = 0;
   out_7842334617962062267[236] = 0;
   out_7842334617962062267[237] = 0;
   out_7842334617962062267[238] = 0;
   out_7842334617962062267[239] = 0;
   out_7842334617962062267[240] = 0;
   out_7842334617962062267[241] = 0;
   out_7842334617962062267[242] = 0;
   out_7842334617962062267[243] = 0;
   out_7842334617962062267[244] = 0;
   out_7842334617962062267[245] = 0;
   out_7842334617962062267[246] = 0;
   out_7842334617962062267[247] = 1;
   out_7842334617962062267[248] = 0;
   out_7842334617962062267[249] = 0;
   out_7842334617962062267[250] = 0;
   out_7842334617962062267[251] = 0;
   out_7842334617962062267[252] = 0;
   out_7842334617962062267[253] = 0;
   out_7842334617962062267[254] = 0;
   out_7842334617962062267[255] = 0;
   out_7842334617962062267[256] = 0;
   out_7842334617962062267[257] = 0;
   out_7842334617962062267[258] = 0;
   out_7842334617962062267[259] = 0;
   out_7842334617962062267[260] = 0;
   out_7842334617962062267[261] = 0;
   out_7842334617962062267[262] = 0;
   out_7842334617962062267[263] = 0;
   out_7842334617962062267[264] = 0;
   out_7842334617962062267[265] = 0;
   out_7842334617962062267[266] = 1;
   out_7842334617962062267[267] = 0;
   out_7842334617962062267[268] = 0;
   out_7842334617962062267[269] = 0;
   out_7842334617962062267[270] = 0;
   out_7842334617962062267[271] = 0;
   out_7842334617962062267[272] = 0;
   out_7842334617962062267[273] = 0;
   out_7842334617962062267[274] = 0;
   out_7842334617962062267[275] = 0;
   out_7842334617962062267[276] = 0;
   out_7842334617962062267[277] = 0;
   out_7842334617962062267[278] = 0;
   out_7842334617962062267[279] = 0;
   out_7842334617962062267[280] = 0;
   out_7842334617962062267[281] = 0;
   out_7842334617962062267[282] = 0;
   out_7842334617962062267[283] = 0;
   out_7842334617962062267[284] = 0;
   out_7842334617962062267[285] = 1;
   out_7842334617962062267[286] = 0;
   out_7842334617962062267[287] = 0;
   out_7842334617962062267[288] = 0;
   out_7842334617962062267[289] = 0;
   out_7842334617962062267[290] = 0;
   out_7842334617962062267[291] = 0;
   out_7842334617962062267[292] = 0;
   out_7842334617962062267[293] = 0;
   out_7842334617962062267[294] = 0;
   out_7842334617962062267[295] = 0;
   out_7842334617962062267[296] = 0;
   out_7842334617962062267[297] = 0;
   out_7842334617962062267[298] = 0;
   out_7842334617962062267[299] = 0;
   out_7842334617962062267[300] = 0;
   out_7842334617962062267[301] = 0;
   out_7842334617962062267[302] = 0;
   out_7842334617962062267[303] = 0;
   out_7842334617962062267[304] = 1;
   out_7842334617962062267[305] = 0;
   out_7842334617962062267[306] = 0;
   out_7842334617962062267[307] = 0;
   out_7842334617962062267[308] = 0;
   out_7842334617962062267[309] = 0;
   out_7842334617962062267[310] = 0;
   out_7842334617962062267[311] = 0;
   out_7842334617962062267[312] = 0;
   out_7842334617962062267[313] = 0;
   out_7842334617962062267[314] = 0;
   out_7842334617962062267[315] = 0;
   out_7842334617962062267[316] = 0;
   out_7842334617962062267[317] = 0;
   out_7842334617962062267[318] = 0;
   out_7842334617962062267[319] = 0;
   out_7842334617962062267[320] = 0;
   out_7842334617962062267[321] = 0;
   out_7842334617962062267[322] = 0;
   out_7842334617962062267[323] = 1;
}
void h_4(double *state, double *unused, double *out_8161452347818855726) {
   out_8161452347818855726[0] = state[6] + state[9];
   out_8161452347818855726[1] = state[7] + state[10];
   out_8161452347818855726[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7761974523197687256) {
   out_7761974523197687256[0] = 0;
   out_7761974523197687256[1] = 0;
   out_7761974523197687256[2] = 0;
   out_7761974523197687256[3] = 0;
   out_7761974523197687256[4] = 0;
   out_7761974523197687256[5] = 0;
   out_7761974523197687256[6] = 1;
   out_7761974523197687256[7] = 0;
   out_7761974523197687256[8] = 0;
   out_7761974523197687256[9] = 1;
   out_7761974523197687256[10] = 0;
   out_7761974523197687256[11] = 0;
   out_7761974523197687256[12] = 0;
   out_7761974523197687256[13] = 0;
   out_7761974523197687256[14] = 0;
   out_7761974523197687256[15] = 0;
   out_7761974523197687256[16] = 0;
   out_7761974523197687256[17] = 0;
   out_7761974523197687256[18] = 0;
   out_7761974523197687256[19] = 0;
   out_7761974523197687256[20] = 0;
   out_7761974523197687256[21] = 0;
   out_7761974523197687256[22] = 0;
   out_7761974523197687256[23] = 0;
   out_7761974523197687256[24] = 0;
   out_7761974523197687256[25] = 1;
   out_7761974523197687256[26] = 0;
   out_7761974523197687256[27] = 0;
   out_7761974523197687256[28] = 1;
   out_7761974523197687256[29] = 0;
   out_7761974523197687256[30] = 0;
   out_7761974523197687256[31] = 0;
   out_7761974523197687256[32] = 0;
   out_7761974523197687256[33] = 0;
   out_7761974523197687256[34] = 0;
   out_7761974523197687256[35] = 0;
   out_7761974523197687256[36] = 0;
   out_7761974523197687256[37] = 0;
   out_7761974523197687256[38] = 0;
   out_7761974523197687256[39] = 0;
   out_7761974523197687256[40] = 0;
   out_7761974523197687256[41] = 0;
   out_7761974523197687256[42] = 0;
   out_7761974523197687256[43] = 0;
   out_7761974523197687256[44] = 1;
   out_7761974523197687256[45] = 0;
   out_7761974523197687256[46] = 0;
   out_7761974523197687256[47] = 1;
   out_7761974523197687256[48] = 0;
   out_7761974523197687256[49] = 0;
   out_7761974523197687256[50] = 0;
   out_7761974523197687256[51] = 0;
   out_7761974523197687256[52] = 0;
   out_7761974523197687256[53] = 0;
}
void h_10(double *state, double *unused, double *out_5436618567726169234) {
   out_5436618567726169234[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_5436618567726169234[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_5436618567726169234[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_7737771249214637356) {
   out_7737771249214637356[0] = 0;
   out_7737771249214637356[1] = 9.8100000000000005*cos(state[1]);
   out_7737771249214637356[2] = 0;
   out_7737771249214637356[3] = 0;
   out_7737771249214637356[4] = -state[8];
   out_7737771249214637356[5] = state[7];
   out_7737771249214637356[6] = 0;
   out_7737771249214637356[7] = state[5];
   out_7737771249214637356[8] = -state[4];
   out_7737771249214637356[9] = 0;
   out_7737771249214637356[10] = 0;
   out_7737771249214637356[11] = 0;
   out_7737771249214637356[12] = 1;
   out_7737771249214637356[13] = 0;
   out_7737771249214637356[14] = 0;
   out_7737771249214637356[15] = 1;
   out_7737771249214637356[16] = 0;
   out_7737771249214637356[17] = 0;
   out_7737771249214637356[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_7737771249214637356[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_7737771249214637356[20] = 0;
   out_7737771249214637356[21] = state[8];
   out_7737771249214637356[22] = 0;
   out_7737771249214637356[23] = -state[6];
   out_7737771249214637356[24] = -state[5];
   out_7737771249214637356[25] = 0;
   out_7737771249214637356[26] = state[3];
   out_7737771249214637356[27] = 0;
   out_7737771249214637356[28] = 0;
   out_7737771249214637356[29] = 0;
   out_7737771249214637356[30] = 0;
   out_7737771249214637356[31] = 1;
   out_7737771249214637356[32] = 0;
   out_7737771249214637356[33] = 0;
   out_7737771249214637356[34] = 1;
   out_7737771249214637356[35] = 0;
   out_7737771249214637356[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_7737771249214637356[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_7737771249214637356[38] = 0;
   out_7737771249214637356[39] = -state[7];
   out_7737771249214637356[40] = state[6];
   out_7737771249214637356[41] = 0;
   out_7737771249214637356[42] = state[4];
   out_7737771249214637356[43] = -state[3];
   out_7737771249214637356[44] = 0;
   out_7737771249214637356[45] = 0;
   out_7737771249214637356[46] = 0;
   out_7737771249214637356[47] = 0;
   out_7737771249214637356[48] = 0;
   out_7737771249214637356[49] = 0;
   out_7737771249214637356[50] = 1;
   out_7737771249214637356[51] = 0;
   out_7737771249214637356[52] = 0;
   out_7737771249214637356[53] = 1;
}
void h_13(double *state, double *unused, double *out_4897505890852390831) {
   out_4897505890852390831[0] = state[3];
   out_4897505890852390831[1] = state[4];
   out_4897505890852390831[2] = state[5];
}
void H_13(double *state, double *unused, double *out_7472495725179531559) {
   out_7472495725179531559[0] = 0;
   out_7472495725179531559[1] = 0;
   out_7472495725179531559[2] = 0;
   out_7472495725179531559[3] = 1;
   out_7472495725179531559[4] = 0;
   out_7472495725179531559[5] = 0;
   out_7472495725179531559[6] = 0;
   out_7472495725179531559[7] = 0;
   out_7472495725179531559[8] = 0;
   out_7472495725179531559[9] = 0;
   out_7472495725179531559[10] = 0;
   out_7472495725179531559[11] = 0;
   out_7472495725179531559[12] = 0;
   out_7472495725179531559[13] = 0;
   out_7472495725179531559[14] = 0;
   out_7472495725179531559[15] = 0;
   out_7472495725179531559[16] = 0;
   out_7472495725179531559[17] = 0;
   out_7472495725179531559[18] = 0;
   out_7472495725179531559[19] = 0;
   out_7472495725179531559[20] = 0;
   out_7472495725179531559[21] = 0;
   out_7472495725179531559[22] = 1;
   out_7472495725179531559[23] = 0;
   out_7472495725179531559[24] = 0;
   out_7472495725179531559[25] = 0;
   out_7472495725179531559[26] = 0;
   out_7472495725179531559[27] = 0;
   out_7472495725179531559[28] = 0;
   out_7472495725179531559[29] = 0;
   out_7472495725179531559[30] = 0;
   out_7472495725179531559[31] = 0;
   out_7472495725179531559[32] = 0;
   out_7472495725179531559[33] = 0;
   out_7472495725179531559[34] = 0;
   out_7472495725179531559[35] = 0;
   out_7472495725179531559[36] = 0;
   out_7472495725179531559[37] = 0;
   out_7472495725179531559[38] = 0;
   out_7472495725179531559[39] = 0;
   out_7472495725179531559[40] = 0;
   out_7472495725179531559[41] = 1;
   out_7472495725179531559[42] = 0;
   out_7472495725179531559[43] = 0;
   out_7472495725179531559[44] = 0;
   out_7472495725179531559[45] = 0;
   out_7472495725179531559[46] = 0;
   out_7472495725179531559[47] = 0;
   out_7472495725179531559[48] = 0;
   out_7472495725179531559[49] = 0;
   out_7472495725179531559[50] = 0;
   out_7472495725179531559[51] = 0;
   out_7472495725179531559[52] = 0;
   out_7472495725179531559[53] = 0;
}
void h_14(double *state, double *unused, double *out_5083935946789046305) {
   out_5083935946789046305[0] = state[6];
   out_5083935946789046305[1] = state[7];
   out_5083935946789046305[2] = state[8];
}
void H_14(double *state, double *unused, double *out_6721528694172379831) {
   out_6721528694172379831[0] = 0;
   out_6721528694172379831[1] = 0;
   out_6721528694172379831[2] = 0;
   out_6721528694172379831[3] = 0;
   out_6721528694172379831[4] = 0;
   out_6721528694172379831[5] = 0;
   out_6721528694172379831[6] = 1;
   out_6721528694172379831[7] = 0;
   out_6721528694172379831[8] = 0;
   out_6721528694172379831[9] = 0;
   out_6721528694172379831[10] = 0;
   out_6721528694172379831[11] = 0;
   out_6721528694172379831[12] = 0;
   out_6721528694172379831[13] = 0;
   out_6721528694172379831[14] = 0;
   out_6721528694172379831[15] = 0;
   out_6721528694172379831[16] = 0;
   out_6721528694172379831[17] = 0;
   out_6721528694172379831[18] = 0;
   out_6721528694172379831[19] = 0;
   out_6721528694172379831[20] = 0;
   out_6721528694172379831[21] = 0;
   out_6721528694172379831[22] = 0;
   out_6721528694172379831[23] = 0;
   out_6721528694172379831[24] = 0;
   out_6721528694172379831[25] = 1;
   out_6721528694172379831[26] = 0;
   out_6721528694172379831[27] = 0;
   out_6721528694172379831[28] = 0;
   out_6721528694172379831[29] = 0;
   out_6721528694172379831[30] = 0;
   out_6721528694172379831[31] = 0;
   out_6721528694172379831[32] = 0;
   out_6721528694172379831[33] = 0;
   out_6721528694172379831[34] = 0;
   out_6721528694172379831[35] = 0;
   out_6721528694172379831[36] = 0;
   out_6721528694172379831[37] = 0;
   out_6721528694172379831[38] = 0;
   out_6721528694172379831[39] = 0;
   out_6721528694172379831[40] = 0;
   out_6721528694172379831[41] = 0;
   out_6721528694172379831[42] = 0;
   out_6721528694172379831[43] = 0;
   out_6721528694172379831[44] = 1;
   out_6721528694172379831[45] = 0;
   out_6721528694172379831[46] = 0;
   out_6721528694172379831[47] = 0;
   out_6721528694172379831[48] = 0;
   out_6721528694172379831[49] = 0;
   out_6721528694172379831[50] = 0;
   out_6721528694172379831[51] = 0;
   out_6721528694172379831[52] = 0;
   out_6721528694172379831[53] = 0;
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

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_2560052200344032530) {
  err_fun(nom_x, delta_x, out_2560052200344032530);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6680368481401804372) {
  inv_err_fun(nom_x, true_x, out_6680368481401804372);
}
void pose_H_mod_fun(double *state, double *out_3134495626797744797) {
  H_mod_fun(state, out_3134495626797744797);
}
void pose_f_fun(double *state, double dt, double *out_3474868352977604468) {
  f_fun(state,  dt, out_3474868352977604468);
}
void pose_F_fun(double *state, double dt, double *out_7842334617962062267) {
  F_fun(state,  dt, out_7842334617962062267);
}
void pose_h_4(double *state, double *unused, double *out_8161452347818855726) {
  h_4(state, unused, out_8161452347818855726);
}
void pose_H_4(double *state, double *unused, double *out_7761974523197687256) {
  H_4(state, unused, out_7761974523197687256);
}
void pose_h_10(double *state, double *unused, double *out_5436618567726169234) {
  h_10(state, unused, out_5436618567726169234);
}
void pose_H_10(double *state, double *unused, double *out_7737771249214637356) {
  H_10(state, unused, out_7737771249214637356);
}
void pose_h_13(double *state, double *unused, double *out_4897505890852390831) {
  h_13(state, unused, out_4897505890852390831);
}
void pose_H_13(double *state, double *unused, double *out_7472495725179531559) {
  H_13(state, unused, out_7472495725179531559);
}
void pose_h_14(double *state, double *unused, double *out_5083935946789046305) {
  h_14(state, unused, out_5083935946789046305);
}
void pose_H_14(double *state, double *unused, double *out_6721528694172379831) {
  H_14(state, unused, out_6721528694172379831);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
