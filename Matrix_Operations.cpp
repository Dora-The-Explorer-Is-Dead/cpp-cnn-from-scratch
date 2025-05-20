#include "..\headers\Matrix_Operations.hpp"

vector<vector<double>> Mat_Mult(vector<vector<double>> a, vector<vector<double>> b) {  // no. of columns for a = no. of rows for b
    if (a.empty() || b.empty()) throw invalid_argument("One or both input matrices are empty.");

    if (a[0].size() != b.size()) throw invalid_argument("Matrix dimensions incompatible for multiplication");

    vector<vector<double>> output(a.size(), vector<double>(b[0].size(), 0.0)); // initialising output matrix - dimensions: no. of rows for a x no. of columns for b

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b[0].size(); j++) {
            for (int k = 0; k < b.size(); k++) { // or a[0].size()
                output[i][j] += a[i][k] * b[k][j];
            } 
        }
    }
    
    return output;
}

vector<vector<double>> broadcast(vector<vector<double>> a, int col) { // to adjust dimensions to allow for Mat_Add
    vector<vector<double>> output(a.size(), vector<double>(col, 0.0));

    if (a.empty() || a[0].size() != 1) throw invalid_argument("Input must have exactly one column.");

    if (col <= 0) throw invalid_argument("Broadcast column count must be greater than 0.");

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < col; j++) {
            output[i][j] = a[i][0];
        }
    }

    return output;
}

vector<vector<double>> Mat_Add(vector<vector<double>> a, vector<vector<double>> b) { // a and b must have the exact same dimensions
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));

    if (a.empty() || b.empty()) throw invalid_argument("Input matrices must not be empty.");

    if (a.size() != b.size() || a[0].size() != b[0].size()) throw invalid_argument("Matrix dimensions must match.");

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            output[i][j] = a[i][j] + b[i][j];
        }
    }

    return output;
}

vector<vector<double>> Mat_Sub(vector<vector<double>> a, vector<vector<double>> b) { // a and b must have the exact same dimensions
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));

    if (a.empty() || b.empty()) throw invalid_argument("Input matrices must not be empty.");

    if (a.size() != b.size() || a[0].size() != b[0].size()) throw invalid_argument("Matrix dimensions must match.");

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            output[i][j] = a[i][j] - b[i][j];
        }
    }

    return output;
}

vector<vector<double>> Mat_Div(vector<vector<double>> a, vector<vector<double>> b) { // a and b must have the exact same dimensions
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));

    if (a.empty() || b.empty()) throw invalid_argument("Input matrices must not be empty.");

    if (a.size() != b.size() || a[0].size() != b[0].size()) throw invalid_argument("Matrix dimensions must match.");

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            if (b[i][j] == 0.0) throw runtime_error("Division by zero in matrix division.");
            output[i][j] = a[i][j] / b[i][j];
        }
    }

    return output;
}

vector<vector<double>> Mult(vector<vector<double>> a, vector<vector<double>> b) { // a and b must have the exact same dimensions
    vector<vector<double>> output(a.size(), vector<double>(a[0].size(), 0.0));

    if (a.empty() || b.empty()) throw invalid_argument("Input matrices must not be empty.");

    if (a.size() != b.size() || a[0].size() != b[0].size()) throw invalid_argument("Matrix dimensions must match.");

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            output[i][j] = a[i][j] * b[i][j];
        }
    }

    return output;
}

vector<vector<double>> transpose(vector<double> a) {
    vector<vector<double>> output(a.size(), vector<double>(1, 0.0));

    for (int i = 0; i < a.size(); i++) {
        output[i][0] = a[i];
    }
    return output;
}

vector<vector<double>> transpose(vector<vector<double>> a) {
    vector<vector<double>> output(a[0].size(), vector<double>(a.size(), 0.0));

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            output[j][i] = a[i][j];
        }
    }
    return output;
}

vector<vector<double>> max(vector<vector<double>> a, int axis) {
    if (axis != 0 && axis != 1) throw invalid_argument("Axis must be 0 (columns) or 1 (rows).");

    if (axis == 1) { // to check for the max element in each row. resultant matrix dimensions: a.size() x 1
        vector<vector<double>> output(a.size(), vector<double>(1, 0.0));
        for (int i = 0; i < a.size(); i++) {
            double max = a[i][0];
            for (int j = 1; j < a[0].size(); j++) {
                if (a[i][j] > max) {
                    max = a[i][j];
                }
            }
            output[i][0] = max;
        }
        return output;
    } 

    if (axis == 0) { // to check for the max element in each col. resultant matrix dimensions: 1 x a[0].size()
        vector<vector<double>> output(1, vector<double>(a[0].size(), 0.0));
        for (int j = 0; j < a[0].size(); j++) {
            double max = a[0][j];
            for (int i = 0; i < a.size(); i++) {
                if (a[i][j] > max) {
                    max = a[i][j];
                }
            }
            output[0][j] = max;
        }
        return output;
    }
}

vector<vector<double>> sum(vector<vector<double>> a, int axis) { 
    if (axis != 0 && axis != 1) throw invalid_argument("Axis must be 0 (columns) or 1 (rows).");

    if (axis == 1) { // for summing every element in a row
        vector<vector<double>> output(a.size(), vector<double>(1, 0.0));
        for (int i = 0; i < a.size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < a[0].size(); j++) {
                sum += a[i][j];
            }
            output[i][0] = sum;
        }
        return output;
    }    

    if (axis == 0) { // for summing every element in a column
        vector<vector<double>> output(1, vector<double>(a[0].size(), 0.0));
        for (int j = 0; j < a[0].size(); j++) {
            double sum = 0.0;
            for (int i = 0; i < a.size(); i++) {
                sum += a[i][j];
            }
            output[0][j] = sum;
        }
        return output;
    }
}