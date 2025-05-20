#include "..\headers\Max_Pooling_Layer.hpp"

int Max_Pooling_Layer::calculate_padding(int I, int K, int S) {
    int O = ceil(static_cast<float>(I) / S);  // Desired output size
    int padding = max(0, (S * (O - 1) + K - I) / 2);
    return padding;
}

vector<vector<double>> Max_Pooling_Layer::do_padding(int col_padding, int row_padding, const vector<vector<double>>& image) {
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

Max_Pooling_Layer::Max_Pooling_Layer(int r, int c, int s, const vector<vector<vector<vector<double>>>>& img) : rows(r), cols(c), stride(s), images(img) {
    /* pool.resize(rows);
    for (int i = 0; i < rows; i++) {
        pool[i].resize(cols, 0.0);
    } */
    outputs.resize(images.size());
} 

vector<vector<vector<vector<double>>>> Max_Pooling_Layer::forward() {
    outputs.resize(images.size());
    max_indices.resize(images.size());

    for (int img_idx = 0; img_idx < images.size(); img_idx++) {
        int channels = images[img_idx].size();
        outputs[img_idx].resize(channels);
        max_indices[img_idx].resize(channels);

        for (int ch = 0; ch < channels; ch++) {
            vector<vector<double>> image = images[img_idx][ch];

            int col_padding = calculate_padding(image[0].size(), cols, stride);
            int row_padding = calculate_padding(image.size(), rows, stride);

            vector<vector<double>> padded_image = do_padding(col_padding, row_padding, image);

            int output_rows = ((padded_image.size() - rows) / stride) + 1;
            int output_cols = ((padded_image[0].size() - cols) / stride) + 1;

            outputs[img_idx][ch].resize(output_rows, vector<double>(output_cols, 0.0));
            max_indices[img_idx][ch].resize(output_rows, vector<pair<int, int>>(output_cols));

            for (int i = 0; i < output_rows; i++) {
                for (int j = 0; j < output_cols; j++) {
                    double max_val = -INFINITY;
                    pair<int, int> max_pos = {0, 0};

                    for (int m = 0; m < rows; m++) {
                        for (int n = 0; n < cols; n++) {
                            int r = i * stride + m;
                            int c = j * stride + n;
                            double val = padded_image[r][c];
                            if (val > max_val) {
                                max_val = val;
                                max_pos = {r, c};
                            }
                        }
                    }

                    outputs[img_idx][ch][i][j] = max_val;
                    max_indices[img_idx][ch][i][j] = max_pos;
                }
            }
        }
    }

    return outputs;
}


vector<vector<vector<vector<double>>>> Max_Pooling_Layer::backward(const vector<vector<vector<vector<double>>>>& dA) {
    vector<vector<vector<vector<double>>>> dinputs(images.size());

    for (int img_idx = 0; img_idx < images.size(); img_idx++) {
        int channels = images[img_idx].size();
        dinputs[img_idx].resize(channels);

        for (int ch = 0; ch < channels; ch++) {
            int padded_rows = images[img_idx][ch].size();
            int padded_cols = images[img_idx][ch][0].size();

            // Initialize with zeros
            dinputs[img_idx][ch].resize(padded_rows, vector<double>(padded_cols, 0.0));

            for (int i = 0; i < dA[img_idx][ch].size(); ++i) {
                for (int j = 0; j < dA[img_idx][ch][0].size(); ++j) {
                    pair<int, int> max_pos = max_indices[img_idx][ch][i][j];
                    int r = max_pos.first;
                    int c = max_pos.second;

                    dinputs[img_idx][ch][r][c] = dA[img_idx][ch][i][j];
                }
            }
        }
    }

    return dinputs;
}



