#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "utils.h"

int main() {

    const int nips = 256;
    const int nops = 10;

    float rate = 1.0f;
    const float eta = 0.99f;

    const int nhid = 28;
    const int iterations = 16;
    const Data data = build("../semeion.data",nips,nops);
    const NeuralNetwork_Type nn = NNbuild(nips, nhid,nops);

    for(int i=0;i<iterations;i++){

        shuffle(data);
        float error = 0.0f;

        for (int j = 0;j<data.rows;j++) {
            printf("debug 1>\n");

            const float * const in = data.in[j];
            const float * const tg = data.tg[j];
            printf("debug 2>\n");
            error += NNtrain(nn,in,tg,rate);
        }
        printf("debug 3>\n");
        printf("Error %.12f :: learning rate %f\n",(double)error/data.rows,(double)rate);
        printf("debug 4>\n");
        rate *=eta;
        printf("debug 5>\n");
    }

    printf("debug 6>\n");

    NNsave(nn,"../mymodel.nn");
    printf("debug 7>\n");
    NNfree(nn);
    printf("debug 8>\n");

    const NeuralNetwork_Type my_loaded_model = NNload("../mymodel.nn");
    printf("debug 9>\n");
    const float * const in = data.in[0];
    const float * const tg = data.tg[0];
    printf("debug 10>\n");

    const float * const pd = NNpredict(my_loaded_model,in);

    printf("debug 11>\n");

    printf("printing target results: \n");
    NNprint(tg,data.nops);
    printf("printing predicted results: \n");
    NNprint(pd,data.nops);
    NNfree(my_loaded_model);
    dfree(data);
    return 0;
}
