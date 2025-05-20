#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

using Image = vector<vector<vector<double>>>; // [channel][row][col]
using Batch = vector<Image>;                           // [image_index][1][28][28]
using Labels = vector<int>;

void load_mnist_csv(const string& filename, Batch& images, Labels& labels);
