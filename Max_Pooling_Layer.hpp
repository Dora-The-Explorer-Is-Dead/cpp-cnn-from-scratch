#include <vector>
#include <cmath>
using namespace std;

class Max_Pooling_Layer {
private:
    int rows;
    int cols;
    int stride;

    vector<vector<vector<vector<double>>>> images;
    vector<vector<vector<vector<double>>>> outputs;
    vector<vector<vector<vector<pair<int, int>>>>> max_indices;

    int calculate_padding(int I, int K, int S);
    vector<vector<double>> do_padding(int col_padding, int row_padding, const vector<vector<double>>& image);

public:
    Max_Pooling_Layer(int r, int c, int s, const vector<vector<vector<vector<double>>>>& img);

    vector<vector<vector<vector<double>>>> forward();
    vector<vector<vector<vector<double>>>> backward(const vector<vector<vector<vector<double>>>>& dA);

    const vector<vector<vector<vector<pair<int, int>>>>>& get_max_indices() const {
        return max_indices;
    }
};
