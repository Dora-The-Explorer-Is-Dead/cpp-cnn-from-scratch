#include "..\headers\Activation_Softmax.hpp"

void Activation_Softmax::forward(vector<vector<double>> inputs) {
    output = Mat_Div(convert_to_exp(inputs), broadcast(sum(convert_to_exp(inputs), 1), inputs[0].size()));
}
