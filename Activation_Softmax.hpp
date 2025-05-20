#include "..\headers\Matrix_Operations.hpp"
#include "..\headers\Activation_Loss_Helper_Functions.hpp"
#include <cmath>
#include <vector>
using namespace std;

class Activation_Softmax {
private:
    vector<vector<double>> output; // probabilities

public:
    void forward(vector<vector<double>> inputs);
};