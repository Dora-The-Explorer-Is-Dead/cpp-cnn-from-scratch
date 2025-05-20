#include "..\headers\Activation_Softmax_Loss_CategoricalCrossEntropy.hpp"

double Activation_Softmax_Loss_CategoricalCrossentropy::forward(const vector<vector<double>>& inputs, const vector<int>& y_true) {
    // Apply softmax
    output = Mat_Div(convert_to_exp(inputs), broadcast(sum(convert_to_exp(inputs), 1), inputs[0].size()));

    // Store and return loss
    last_loss = calculate_loss(output, y_true);

    // Compute and store accuracy
    last_accuracy = calculate_accuracy(output, y_true);

    return last_loss;
}

void Activation_Softmax_Loss_CategoricalCrossentropy::backward(const vector<int>& y_true) {
    size_t samples = output.size();
    size_t labels = output[0].size();

    dinputs = output;

    for (size_t i = 0; i < samples; ++i) {
        dinputs[i][y_true[i]] -= 1.0;
    }

    // Normalize gradients
    for (size_t i = 0; i < samples; ++i) {
        for (size_t j = 0; j < labels; ++j) {
            dinputs[i][j] /= samples;
        }
    }
}

double Activation_Softmax_Loss_CategoricalCrossentropy::calculate_loss(const vector<vector<double>>& y_pred, const vector<int>& y_true) {
    double loss = 0.0;
    size_t samples = y_pred.size();

    for (size_t i = 0; i < samples; ++i) {
        double correct_confidence = y_pred[i][y_true[i]];
        correct_confidence = max(correct_confidence, 1e-15); 
        loss += -log(correct_confidence);
    }

    return loss / samples;
}

double Activation_Softmax_Loss_CategoricalCrossentropy::calculate_accuracy(const vector<vector<double>>& y_pred, const vector<int>& y_true) {
    size_t correct = 0;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        int pred_label = distance(y_pred[i].begin(), max_element(y_pred[i].begin(), y_pred[i].end()));
        if (pred_label == y_true[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / y_pred.size();
}

double Activation_Softmax_Loss_CategoricalCrossentropy::get_last_loss() const {
    return last_loss;
}

double Activation_Softmax_Loss_CategoricalCrossentropy::get_last_accuracy() const {
    return last_accuracy;
}

