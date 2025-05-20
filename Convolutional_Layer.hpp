#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Convolutional_Layer {
private:
    int rows_no;
    int cols_no;
    int kernels_no; // kernels per filter
    int filters_no;
    int stride;
    int row_padding;
    int col_padding;
    double learning_rate = 0.01; // Default learning rate
    string activation_function;
    vector<vector<vector<vector<double>>>> filters; // [filters][kernels][rows][cols]
    vector<double> biases; // one bias per filter
    vector<vector<vector<vector<double>>>> images;
    vector<vector<vector<vector<double>>>> padded_images;
    vector<vector<vector<vector<double>>>> post_activation_outputs; // A
    vector<vector<vector<vector<double>>>> pre_activation_outputs; // Z
    vector<vector<vector<vector<double>>>> filters_gradients;
    vector<double> biases_gradients;
    vector<vector<vector<vector<double>>>> inputs_gradients;

    int calculate_padding(int I, int K, int S);

    vector<vector<double>> do_padding(int col_padding, int row_padding, vector<vector<double>> image);

    double activate(double val);

    void update_parameters();

public:
    Convolutional_Layer(int r, int c, int k, int f, int s, string a, const vector<vector<vector<vector<double>>>>& i);

    vector<vector<vector<vector<double>>>> forward();

    vector<vector<vector<vector<double>>>> backward(const vector<vector<vector<vector<double>>>>& dA);

    void set_learning_rate(double lr);

    double activation_derivative(double val);
};