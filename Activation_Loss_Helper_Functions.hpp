#include "..\headers\Matrix_Operations.hpp"
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

vector<vector<double>> remove_neg(const vector<vector<double>>& a);

vector<vector<double>> convert_to_exp(const vector<vector<double>>& a);

vector<vector<double>> convert_to_neg_log(const vector<vector<double>>& a);

vector<vector<double>> clip(const vector<vector<double>>& a);

vector<vector<double>> advanced_indexing_simple(const vector<vector<double>>& softmax_outputs, const vector<int>& label_values, const vector<int>& class_targets);

vector<vector<double>> advanced_indexing(const vector<vector<double>>& softmax_outputs, const vector<int>& label_values, const vector<vector<double>>& class_targets);

