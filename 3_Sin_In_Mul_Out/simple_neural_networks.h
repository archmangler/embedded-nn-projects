#ifndef _SIMPLE_NEURAL_NETWORKS_H
#define _SIMPLE_NEURAL_NETWORKS_H
double single_in_single_out(double input, double weight);
double multiple_input_single_output(double * input,double * weight,int LEN);
void single_in_multiple_out_nn(double input_scalar, double * weight_vector, double * output_vector, int LEN);
#endif //_SIMPE_NEURAL_NETWORKS_H
