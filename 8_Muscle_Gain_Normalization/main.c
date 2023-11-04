#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "simple_neural_networks.h"
#define NUM_OF_INPUTS 3
#define Sad 0.9
#define SAD_PREDICTION_IDX 0
#define SICK_PREDICTION_IDX 1
#define ACTIVE_PREDICTION_IDX 2
#define NUM_OF_FEATURES 2
#define NUM_OF_EXAMPLE 3

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

//for brute force learning
double weight = 0.5;
double input = 0.5;
double expected_value = 0.8;
double step_amount = 0.001;

/*HOURS OF WORKOUT DATA*/
double x1[NUM_OF_EXAMPLE] = {2,5,1};
double _x1[NUM_OF_EXAMPLE];

/*Hours of rest data*/
double x2[NUM_OF_EXAMPLE] = {8,5,8};
double _x2[NUM_OF_EXAMPLE];

/*Muscle Gain Data*/
double y[NUM_OF_EXAMPLE] = {200,90,190};
double _y[NUM_OF_EXAMPLE];

int main() {

    normalize_data(x1,_x1,NUM_OF_EXAMPLE);
    normalize_data(x2,_x2,NUM_OF_EXAMPLE);
    normalize_data(y,_y,NUM_OF_EXAMPLE);

    printf("Raw x1 data: \n");
    for(int i=0;i<NUM_OF_EXAMPLE;i++){
        printf("%f",x1[i]);
    }
    printf("\n");
    printf("Normalized x1 data: \n");
    for(int i=0;i<NUM_OF_EXAMPLE;i++){
        printf("%f",_x1[i]);
    }
    printf("\n");

    printf("Raw x2 data: \n");
    for(int i=0;i<NUM_OF_EXAMPLE;i++){
        printf("%f",x2[i]);
    }
    printf("\n");
    printf("Normalized x2 data: \n");
    for(int i=0;i<NUM_OF_EXAMPLE;i++){
        printf("%f",_x2[i]);
    }
    printf("\n");

    printf("Raw y data: \n");
    for(int i=0;i<NUM_OF_EXAMPLE;i++){
        printf("%f",y[i]);
    }
    printf("\n");
    printf("Normalized y data: \n");
    for(int i=0;i<NUM_OF_EXAMPLE;i++){
        printf("%f",_y[i]);
    }
    printf("\n");

    /*
    brute_force_learning(input,weight,expected_value,step_amount,1300);
    hidden_layer_nn(inputs,IN_LEN,HID_LEN,input_to_hidden_weights,OUT_LEN,hidden_to_output_weights,predicted_results);
    printf("Sad prediction : %f \n",predicted_results[SAD_PREDICTION_IDX]);
    printf("Sad error: %f\n", find_error_simple(predicted_results[SAD_PREDICTION_IDX],expected_values[SAD_PREDICTION_IDX]));
    printf("Sick prediction : %f \n",predicted_results[SICK_PREDICTION_IDX]);
    printf("Sick error: %f\n", find_error_simple(predicted_results[SICK_PREDICTION_IDX],expected_values[SICK_PREDICTION_IDX]));
    printf("Active prediction : %f \n",predicted_results[ACTIVE_PREDICTION_IDX]);
    printf("Active error: %f\n", find_error_simple(predicted_results[ACTIVE_PREDICTION_IDX],expected_values[ACTIVE_PREDICTION_IDX]));
*/

    return 0;
}
