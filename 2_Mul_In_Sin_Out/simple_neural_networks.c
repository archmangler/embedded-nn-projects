double single_in_single_out(double input, double weight){
    return (input * weight);
}

double weighted_sum(double * input, double * weight, int LEN){
    double output;
    for(int i=0; i < LEN; i++){
        output += input[i]*weight[i];
    }
    return output;
}

double multiple_input_single_output(double * input, double * weight, int LEN){
    double predicted_value;
    predicted_value = weighted_sum(input, weight, LEN);
    return predicted_value;
}
