#include "..\headers\Convolutional_Layer.hpp"

int Convolutional_Layer::calculate_padding(int I, int K, int S) {
    int O = ceil(I / S);  // Desired output size
    int padding = max(0, (S * (O - 1) + K - I) / 2);
    return padding;
}

vector<vector<double>> Convolutional_Layer::do_padding(int col_padding, int row_padding, vector<vector<double>> image) {
    vector<vector<double>> padded(image.size() + row_padding, vector<double>(image[0].size() + col_padding, 0.0));
    int left = col_padding / 2;
    int right = col_padding - left;
    int top = row_padding / 2;
    int bottom = row_padding - top;
    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[0].size(); j++) {
            padded[top + i][left + j] = image[i][j];
        }
    }
    return padded;
}

double Convolutional_Layer::activate(double val) {
    if (activation_function == "sigmoid") {
        return 1.0 / (1.0 + exp(-val)); // Sigmoid
    } else { // Default to ReLU
        return (val > 0) ? val : 0.0; // ReLU
    }
}

Convolutional_Layer::Convolutional_Layer(int r, int c, int k, int f, int s, string a, const vector<vector<vector<vector<double>>>>& i) : rows_no(r), cols_no(c), kernels_no(k), filters_no(f), stride(s), activation_function(a), images(i) {

    int batch_size = images.size();
    int input_channels = images[0].size();

    filters.resize(filters_no, vector<vector<vector<double>>>(input_channels, vector<vector<double>>(rows_no, vector<double>(cols_no))));

    // Choose initializer based on activation function
    double limit;
    if (activation_function == "relu") {
        limit = sqrt(2.0 / (rows_no * cols_no * input_channels));  // He initialization
    } else if (activation_function == "sigmoid") {
        limit = sqrt(1.0 / (rows_no * cols_no * input_channels));  // Xavier initialization
    } else {
        limit = 0.01;  // Fallback small random range
    }

    // Initialize filters with values in [-limit, limit]
    for (int f_idx = 0; f_idx < filters_no; f_idx++) {
        for (int ch = 0; ch < input_channels; ch++) {
            for (int r_idx = 0; r_idx < rows_no; r_idx++) {
                for (int c_idx = 0; c_idx < cols_no; c_idx++) {
                    filters[f_idx][ch][r_idx][c_idx] = limit * ((double)rand() / RAND_MAX * 2.0 - 1.0);
                }
            }
        }
    }

    // Bias for each filter
    biases.resize(filters_no, 0.0);
}


vector<vector<vector<vector<double>>>> Convolutional_Layer::forward() {
    int batch_size = images.size();
    int channels = images[0].size();
    int input_rows = images[0][0].size();
    int input_cols = images[0][0][0].size();

    row_padding = calculate_padding(input_rows, rows_no, stride);
    col_padding = calculate_padding(input_cols, cols_no, stride);

    int output_rows = ((input_rows + 2 * row_padding - rows_no) / stride) + 1;
    int output_cols = ((input_cols + 2 * col_padding - cols_no) / stride) + 1;

    pre_activation_outputs.resize(batch_size, vector<vector<vector<double>>>(filters_no, vector<vector<double>>(output_rows, vector<double>(output_cols, 0.0))));
    post_activation_outputs = pre_activation_outputs;

    padded_images.resize(batch_size, vector<vector<vector<double>>>(channels, vector<vector<double>>()));

    for (int b = 0; b < batch_size; ++b)
        for (int c = 0; c < channels; ++c)
            padded_images[b][c] = do_padding(col_padding, row_padding, images[b][c]);

    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < filters_no; ++f) {
            for (int i = 0; i < output_rows; ++i) {
                for (int j = 0; j < output_cols; ++j) {
                    double sum = 0.0;
                    for (int c = 0; c < channels; ++c) {
                        for (int m = 0; m < rows_no; ++m) {
                            for (int n = 0; n < cols_no; ++n) {
                                int img_row = i * stride + m;
                                int img_col = j * stride + n;
                                sum += filters[f][c][m][n] * padded_images[b][c][img_row][img_col];
                            }
                        }
                    }
                    double z = sum + biases[f];
                    pre_activation_outputs[b][f][i][j] = z;
                    post_activation_outputs[b][f][i][j] = activate(z);
                }
            }
        }
    }

    return post_activation_outputs;
}


vector<vector<vector<vector<double>>>> Convolutional_Layer::backward(const vector<vector<vector<vector<double>>>>& dA) {
    int batch_size = dA.size();
    int channels = images[0].size();
    int input_rows = images[0][0].size();
    int input_cols = images[0][0][0].size();
    int output_rows = dA[0][0].size();
    int output_cols = dA[0][0][0].size();

    filters_gradients.assign(filters_no, vector<vector<vector<double>>>(channels, vector<vector<double>>(rows_no, vector<double>(cols_no, 0.0))));
    biases_gradients.assign(filters_no, 0.0);
    inputs_gradients.assign(batch_size, vector<vector<vector<double>>>(channels, vector<vector<double>>(input_rows, vector<double>(input_cols, 0.0))));

    vector<vector<vector<vector<double>>>> padded_inputs_gradients(batch_size, vector<vector<vector<double>>>(channels, vector<vector<double>>(input_rows + 2 * row_padding, vector<double>(input_cols + 2 * col_padding, 0.0))));

    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < filters_no; ++f) {
            for (int i = 0; i < output_rows; ++i) {
                for (int j = 0; j < output_cols; ++j) {
                    double z = pre_activation_outputs[b][f][i][j];
                    double dZ = dA[b][f][i][j] * activation_derivative(z);

                    biases_gradients[f] += dZ;

                    for (int c = 0; c < channels; ++c) {
                        for (int m = 0; m < rows_no; ++m) {
                            for (int n = 0; n < cols_no; ++n) {
                                int img_row = i * stride + m;
                                int img_col = j * stride + n;

                                filters_gradients[f][c][m][n] += dZ * padded_images[b][c][img_row][img_col];
                                padded_inputs_gradients[b][c][img_row][img_col] += filters[f][c][m][n] * dZ;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < input_rows; ++i) {
                for (int j = 0; j < input_cols; ++j) {
                    inputs_gradients[b][c][i][j] = padded_inputs_gradients[b][c][i + row_padding / 2][j + col_padding / 2];
                }
            }
        }
    }

    update_parameters();
    return inputs_gradients;
}


void Convolutional_Layer::set_learning_rate(double lr) {
    learning_rate = lr;
}

void Convolutional_Layer::update_parameters() {
    for (int f = 0; f < filters_no; f++) {
        for (int k = 0; k < kernels_no; k++) {
            for (int i = 0; i < rows_no; i++) {
                for (int j = 0; j < cols_no; j++) {
                    filters[f][k][i][j] -= learning_rate * filters_gradients[f][k][i][j];
                }
            }
        }
        biases[f] -= learning_rate * biases_gradients[f];
    }
}

double Convolutional_Layer::activation_derivative(double val) {
    if (activation_function == "sigmoid") {
        double sig = 1.0 / (1.0 + exp(-val));
        return sig * (1.0 - sig);
    } else { // ReLU
        return (val > 0) ? 1.0 : 0.0;
    }
}


