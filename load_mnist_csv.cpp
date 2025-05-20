#include "..\headers\load_mnist_csv.hpp"

using Image = vector<vector<vector<double>>>; // [channel][row][col]
using Batch = vector<Image>;                           // [image_index][1][28][28]
using Labels = vector<int>; 

void load_mnist_csv(const string& filename, Batch& images, Labels& labels) {
    ifstream file(filename);
    string line;

    bool skip_header = true;
    while (getline(file, line)) {
        if (skip_header) {
            skip_header = false;
            continue; // skip first row (header)
        }

        stringstream ss(line);
        string value;
        vector<double> pixels;
        int label;

        // Get label
        getline(ss, value, ',');
        label = stoi(value);
        labels.push_back(label);

        // Get 784 pixel values
        while (getline(ss, value, ',')) {
            double pixel = stod(value) / 255.0; // Normalize
            pixels.push_back(pixel);
        }

        // Reshape to 1x28x28
        Image img(1, vector<vector<double>>(28, vector<double>(28, 0.0)));
        for (int i = 0; i < 28; ++i)
            for (int j = 0; j < 28; ++j)
                img[0][i][j] = pixels[i * 28 + j];

        images.push_back(img);
    }
}
