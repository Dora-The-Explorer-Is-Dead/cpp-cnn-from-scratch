#include "..\headers\Activation_Loss_Helper_Functions.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

class Loss_CategoricalCrossEntropy {
public:
    vector<vector<double>> forward(const vector<vector<double>>& y_pred, const vector<vector<int>>& y_true);

    vector<vector<double>> Loss_CategoricalCrossEntropy::backward(const vector<vector<double>>& y_pred, const vector<vector<int>>& y_true);
};
