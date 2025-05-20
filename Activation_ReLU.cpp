#include "..\headers\Activation_ReLU.hpp"

void Activation_ReLU::forward(vector<vector<double>> inputs) {
    output = remove_neg(inputs);
}
