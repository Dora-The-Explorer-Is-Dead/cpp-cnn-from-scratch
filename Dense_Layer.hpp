#pragma once
#include "..\headers\Matrix_Operations.hpp"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace std;

class Dense_Layer {
private:
    int neurons;
    int no_inputs;
    double learning_rate = 0.01; // Default learning rate

    vector<vector<double>> inputs;
    vector<vector<double>> weights;
    vector<vector<double>> biases;
    vector<vector<double>> output;
    vector<vector<double>> d_inputs;
    vector<vector<double>> d_weights;
    vector<vector<double>> d_biases;

    void update_parameters();

public:
    Dense_Layer(int n = 0, int i = 0);

    void set_learning_rate(double lr);

    void forward(vector<vector<double>> inputs);
    void backward(vector<vector<double>> d_out);

    vector<vector<double>> get_output() const;
    vector<vector<double>> get_d_inputs() const;
};
