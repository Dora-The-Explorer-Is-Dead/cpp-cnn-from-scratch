#include "..\headers\Loss_CategoricalCrossEntropy.hpp"

vector<vector<double>> Loss_CategoricalCrossEntropy::forward(const vector<vector<double>>& y_pred, const vector<vector<int>>& y_true) {
    vector<vector<double>> clipped_y_pred = clip(y_pred);

    vector<vector<double>> losses(y_pred.size(), vector<double>(1, 0.0));

    bool is_one_hot = y_true[0].size() > 1;

    for (int i = 0; i < y_pred.size(); ++i) {
        if (is_one_hot) {
            double loss = 0.0;
            for (int j = 0; j < y_pred[0].size(); j++) {
                loss += y_true[i][j] * log(clipped_y_pred[i][j]);
            }
            losses[i][0] = -loss;
        } else {
            int correct_class = y_true[i][0];
            losses[i][0] = -log(clipped_y_pred[i][correct_class]);
        }
    }

    return losses;
}

vector<vector<double>> Loss_CategoricalCrossEntropy::backward(const vector<vector<double>>& y_pred, const vector<vector<int>>& y_true) {
    int samples = y_pred.size();
    int labels = y_pred[0].size();

    vector<vector<double>> d_inputs(samples, vector<double>(labels, 0.0));

    bool is_one_hot = y_true[0].size() > 1;

    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < labels; ++j) {
            if (is_one_hot) {
                d_inputs[i][j] = (y_pred[i][j] - y_true[i][j]) / samples;
            } else {
                int correct_class = y_true[i][0];
                d_inputs[i][j] = j == correct_class ? (y_pred[i][j] - 1.0) / samples : y_pred[i][j] / samples;
            }
        }
    }

    return d_inputs;
}

