#include "nn.h"
#include "utils.h"


int main(){


const int nips =256;
const int nops =10;


const Data inference  = build("test.data", nips, nops);
const NeuralNetwork_Type loaded_model =  NNload("mymodel2.nn");

const float * const in = inference.in[1];
const float * const pd = NNpredict(loaded_model,in);


NNprint(pd,inference.nops);
NNfree(loaded_model);
dfree(inference);

return 0;


}
