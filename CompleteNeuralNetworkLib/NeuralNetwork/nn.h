//
// Created by Traiano Welcome on 28/10/23.
//

#ifndef NEURALNETWORK_NN_H
#define NEURALNETWORK_NN_H

typedef struct
{
    float  *w; /*All the weights*/
    float *x; /*hidden layer to output layer weights*/
    float *b; /*biases*/
    float *h; /*hidden layer*/
    float *o; /*output layer*/
    int nb; /*number of biases*/
    int nw; /*number of weights*/
    int nips; /*number of inputs*/
    int nhid; /*number of hidden neurons*/
    int nops; /*number of outputs*/
} NeuralNetwork_Type;

float * NNpredict(const NeuralNetwork_Type nn, const float * in);

NeuralNetwork_Type NNbuild(int nips, int nhid, int nops);

float NNtrain(const NeuralNetwork_Type nn, const float * in, const float *tg, float rate);

void NNsave(const NeuralNetwork_Type nn, const char * path);

NeuralNetwork_Type NNload(const char * path);

void NNprint(const float * arr, const int size);

void NNfree(const NeuralNetwork_Type nn);

static float frand();

#endif //NEURALNETWORK_NN_H
