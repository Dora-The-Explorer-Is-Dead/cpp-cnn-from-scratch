#include "..\headers\Flatten_Layer.hpp"

vector<vector<double>> Flatten_Layer::forward(const vector<vector<vector<vector<double>>>>& inputs) {
    batch_size = inputs.size();
    channels = inputs[0].size();
    height = inputs[0][0].size();
    width = inputs[0][0][0].size();

    vector<vector<double>> output(batch_size, vector<double>(channels * height * width));

    for (int i = 0; i < batch_size; i++) {
        int flat_index = 0;
        for (int j = 0; j < channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    output[i][flat_index++] = inputs[i][j][k][l];
                }
            }
        }
    }

    return output;
}

vector<vector<vector<vector<double>>>> Flatten_Layer::backward(const vector<vector<double>>& d_out) {
    vector<vector<vector<vector<double>>>> d_input(batch_size, vector<vector<vector<double>>>(channels, vector<vector<double>>(height, vector<double>(width))));

    for (int i = 0; i < batch_size; i++) {
        int flat_index = 0;
        for (int j = 0; j < channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    d_input[i][j][k][l] = d_out[i][flat_index++];
                }
            }
        }
    }

    return d_input;
}
