#ifndef _SIMPLE_NEURAL_NETWORKS_H
#define _SIMPLE_NEURAL_NETWORKS_H
double single_in_single_out(double input, double weight);
double multiple_input_single_output(double * input, double * weight, int LEN);
void multiple_input_multiple_output_nn (double * input_vector, int INPUT_LEN, double * output_vector, int OUTPUT_LEN, double weight_matrix[OUTPUT_LEN][INPUT_LEN]);
void hidden_layer_nn(double * input_vector,
                     int INPUT_LEN,
                     int HIDDEN_LEN,
                     double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
                     int OUTPUT_LEN,
                     double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN],
                     double * output_vector);

double find_error(double input, double weight, double expected_value);
double find_error_simple(double yhat,double y);

#endif //SIMPE_NEURAL_NETWORKS_H
