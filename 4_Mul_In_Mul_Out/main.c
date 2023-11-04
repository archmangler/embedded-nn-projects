#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"
#define NUM_OF_INPUTS 3
#define Sad 0.9
#define SAD_PREDICTION_IDX 0
#define SICK_PREDICTION_IDX 1
#define ACTIVE_PREDICTION_IDX 2

#define IN_LEN 3
#define OUT_LEN 3

double predicted_results[3];

double weights[OUT_LEN][IN_LEN] = {

        {-2,9.5,2.01},
        {-0.8,7.2,6.3},
        {-0.5,0.455,0.9}
};

double inputs[IN_LEN] = {30, 87, 110};

int main() {

    multiple_input_multiple_output_nn(inputs, IN_LEN,predicted_results, OUT_LEN, weights);

    printf("Sad prediction : %f \n",predicted_results[SAD_PREDICTION_IDX]);
    printf("Sick prediction : %f \n",predicted_results[SICK_PREDICTION_IDX]);
    printf("Active prediction : %f \n",predicted_results[ACTIVE_PREDICTION_IDX]);

    return 0;
}