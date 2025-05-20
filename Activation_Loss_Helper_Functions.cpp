#include "..\headers\Activation_Loss_Helper_Functions.hpp"

vector<vector<double>> remove_neg(const vector<vector<double>>& a) {
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            if (a[i][j] < 0.0) output[i][j] = 0.0;
            else output[i][j] = a[i][j];
        }
    }
    return output;
}

vector<vector<double>> convert_to_exp(const vector<vector<double>>& a) {
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            output[i][j] = exp(a[i][j]);
        }
    }
    return output;
}

vector<vector<double>> convert_to_neg_log(const vector<vector<double>>& a) {
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            output[i][j] = (a[i][j] > 0.0) ? -1 * log(a[i][j]) : 0.0;  
        }
    }
    return output;
}

vector<vector<double>> clip(const vector<vector<double>>& a) {
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            /* if (a[i][j] < 1e-7) output[i][j] = 1e-7;
            else if (a[i][j] > 1 - 1e-7) output[i][j] = 1 - 1e-7;
            else output[i][j] = a[i][j]; */ 
            output[i][j] = (a[i][j] < 1e-7) ? 1e-7 : (a[i][j] > 1 - 1e-7) ? 1 - 1e-7 : a[i][j];
        }
    }
    return output;
}

vector<vector<double>> advanced_indexing_simple(const vector<vector<double>>& softmax_outputs, const vector<int>& label_values, const vector<int>& class_targets) {
    vector<vector<double>> output(label_values.size(), vector<double>(1, 0.0));
    for (int i = 0; i < label_values.size(); i++) {
        output[i][0] = softmax_outputs[label_values[i]][class_targets[i]];
    }
    return output;
}

vector<vector<double>> advanced_indexing(const vector<vector<double>>& softmax_outputs, const vector<int>& label_values, const vector<vector<double>>& class_targets) {
    vector<vector<double>> output(label_values.size(), vector<double>(1, 0.0));
    output = sum(Mult(class_targets, softmax_outputs), 1);
    return output;
}



