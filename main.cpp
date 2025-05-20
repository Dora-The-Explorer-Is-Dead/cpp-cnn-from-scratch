#include "..\headers\load_mnist_csv.hpp"
#include "..\headers\Convolutional_Layer.hpp"
#include "..\headers\Max_Pooling_Layer.hpp"
#include "..\headers\Flatten_Layer.hpp"
#include "..\headers\Dense_Layer.hpp"
#include "..\headers\Activation_Softmax_Loss_CategoricalCrossEntropy.hpp"

int main() {
    Batch images;
    Labels labels;

    load_mnist_csv("mnist_train.csv", images, labels);

    // batch_size you want to process at once
    int batch_size = 32;

    // Start index in your images vector
    int start_idx = 0;  // or wherever you want to start

    // Prepare 4D vector: [batch_size][channels][height][width]
    vector<vector<vector<vector<double>>>> input_batch(batch_size, vector<vector<vector<double>>>(1, vector<vector<double>>(28, vector<double>(28, 0.0))));

    // Copy images from your Batch into this 4D vector
    for (int i = 0; i < batch_size; ++i) {
        const Image& img = images[start_idx + i];  // images loaded from CSV

        for (int c = 0; c < 1; ++c) {            // channels = 1 for MNIST grayscale
            for (int r = 0; r < 28; ++r) {
                for (int col = 0; col < 28; ++col) {
                    input_batch[i][c][r][col] = img[c][r][col];
                }
            }
        }
    }

    Convolutional_Layer convolulu1(5, 5, 1, 2, 1, "relu", input_batch);
    vector<vector<vector<vector<double>>>> cl1_output = convolulu1.forward();
    Max_Pooling_Layer max_liverpool1(2, 2, 2, cl1_output);
    vector<vector<vector<vector<double>>>> ml1_output = max_liverpool1.forward();
    Convolutional_Layer convolulu2(3, 3, 2, 4, 1, "sigmoid", ml1_output);
    vector<vector<vector<vector<double>>>> cl2_output = convolulu2.forward();
    Max_Pooling_Layer max_liverpool2(2, 2, 2, cl2_output);
    vector<vector<vector<vector<double>>>> ml2_output = max_liverpool2.forward();
    Flatten_Layer falafel;
    vector<vector<double>> f1_output = falafel.forward(ml2_output);
    Dense_Layer dance(f1_output[0].size(), 32);
    dance.forward(f1_output);
    Activation_Softmax_Loss_CategoricalCrossentropy blabla;
    blabla.forward(dance.get_output());


    



}



