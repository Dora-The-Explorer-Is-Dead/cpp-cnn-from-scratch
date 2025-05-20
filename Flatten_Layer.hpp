#include <iostream>
#include <vector>
using namespace std;

class Flatten_Layer {
private:
    vector<vector<double>> output;
    int batch_size;
    int channels;
    int height;
    int width;

public:
    vector<vector<double>> forward(const vector<vector<vector<vector<double>>>>& inputs);

    vector<vector<vector<vector<double>>>> backward(const vector<vector<double>>& d_out);
    
};