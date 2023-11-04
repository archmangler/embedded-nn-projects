#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "simple_neural_networks.h"
#define NUM_OF_INPUTS 3
#define Sad 0.9
#define SAD_PREDICTION_IDX 0
#define SICK_PREDICTION_IDX 1
#define ACTIVE_PREDICTION_IDX 2

#define IN_LEN 3
#define OUT_LEN 3
#define HID_LEN 3

double input_to_hidden_weights[OUT_LEN][IN_LEN] = {
        {-2,9.5,2.01},
        {-0.8,7.2,6.3},
        {-0.5,0.45,0.9}
};

double hidden_to_output_weights[OUT_LEN][IN_LEN] = {
        {-1.0, 1.15, 0.11},
        {-0.18,0.15,-0.01},
        {0.25, -0.25,-0.1}
};

double predicted_results[3];

double weights[OUT_LEN][IN_LEN] = {

        {-2,9.5,2.01},
        {-0.8,7.2,6.3},
        {-0.5,0.455,0.9}
};

double inputs[IN_LEN] = {30, 87, 110};
double expected_values[] = {600,10,-90};

int main() {

    hidden_layer_nn(inputs,IN_LEN,HID_LEN,input_to_hidden_weights,OUT_LEN,hidden_to_output_weights,predicted_results);

    printf("Sad prediction : %f \n",predicted_results[SAD_PREDICTION_IDX]);
    printf("Sad error: %f\n", find_error_simple(predicted_results[SAD_PREDICTION_IDX],expected_values[SAD_PREDICTION_IDX]));

    printf("Sick prediction : %f \n",predicted_results[SICK_PREDICTION_IDX]);
    printf("Sick error: %f\n", find_error_simple(predicted_results[SICK_PREDICTION_IDX],expected_values[SICK_PREDICTION_IDX]));

    printf("Active prediction : %f \n",predicted_results[ACTIVE_PREDICTION_IDX]);
    printf("Active error: %f\n", find_error_simple(predicted_results[ACTIVE_PREDICTION_IDX],expected_values[ACTIVE_PREDICTION_IDX]));

    return 0;
}
