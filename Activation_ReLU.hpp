#include "..\headers\Activation_Loss_Helper_Functions.hpp"
#include <vector>
using namespace std;

class Activation_ReLU {
private:
    vector<vector<double>> output;

public:
    void forward(vector<vector<double>> inputs);
};