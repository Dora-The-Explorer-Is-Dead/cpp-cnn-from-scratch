#include "..\headers\Dense_Layer.hpp"

Dense_Layer::Dense_Layer(int n, int i) : neurons(n), no_inputs(i) {
    srand(time(0));

    // Initialize weights: [no_inputs x neurons]
    weights = vector<vector<double>>(no_inputs, vector<double>(neurons, 0.0));

    // Initialize biases: [1 x neurons]
    biases = vector<vector<double>>(1, vector<double>(neurons, 0.0));

    // He initialization (good for ReLU)
    /* double limit = sqrt(2.0 / no_inputs);
    for (int i = 0; i < no_inputs; i++) {
        for (int j = 0; j < neurons; j++) {
            weights[i][j] = limit * ((double)rand() / RAND_MAX * 2 - 1);  // [-limit, limit]
        }
    } */

    // Xavier initialization
    double limit = sqrt(6.0 / (no_inputs + neurons));
    for (int i = 0; i < no_inputs; i++) {
        for (int j = 0; j < neurons; j++) {
            weights[i][j] = limit * ((double)rand() / RAND_MAX * 2 - 1);  // [-limit, limit]
        }
    }

}

void Dense_Layer::set_learning_rate(double lr) {
    learning_rate = lr;
}

void Dense_Layer::forward(vector<vector<double>> inputs) {
    this->inputs = inputs;

    // Safety check
    if (inputs[0].size() != weights.size())
        throw invalid_argument("Input size does not match weight matrix dimensions.");

    // Mat_Mult(input, weights) → shape: [batch x neurons]
    output = Mat_Add(Mat_Mult(inputs, weights), broadcast(biases, inputs.size()));
}

void Dense_Layer::backward(vector<vector<double>> d_out) {
    if (d_out.size() != inputs.size())
        throw invalid_argument("Gradient size does not match batch size.");

    d_weights = Mat_Mult(transpose(inputs), d_out);           // [inputs x batch] x [batch x neurons] = [inputs x neurons]
    d_biases = sum(d_out, 0);                                  // Sum across batches
    d_inputs = Mat_Mult(d_out, transpose(weights));            // Backprop to previous layer

    update_parameters();
}

void Dense_Layer::update_parameters() {
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            weights[i][j] -= learning_rate * d_weights[i][j];
        }
    }

    for (int i = 0; i < biases[0].size(); i++) {
        biases[0][i] -= learning_rate * d_biases[0][i];
    }
}

vector<vector<double>> Dense_Layer::get_output() const {
    return output;
}

vector<vector<double>> Dense_Layer::get_d_inputs() const {
    return d_inputs;
}
