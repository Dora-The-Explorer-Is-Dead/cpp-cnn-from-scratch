#include "..\headers\Matrix_Operations.hpp"
#include "..\headers\Activation_Loss_Helper_Functions.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>
using namespace std;

class Activation_Softmax_Loss_CategoricalCrossentropy {
public:
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    double last_loss;
    double last_accuracy;

    double forward(const vector<vector<double>>& inputs, const vector<int>& y_true);

    void backward(const vector<int>& y_true);

    double calculate_loss(const vector<vector<double>>& y_pred, const vector<int>& y_true);

    double calculate_accuracy(const vector<vector<double>>& y_pred, const vector<int>& y_true);

    double get_last_loss() const;

    double get_last_accuracy() const;
};
