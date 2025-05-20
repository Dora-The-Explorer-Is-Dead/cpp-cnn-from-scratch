#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

vector<vector<double>> Mat_Mult(vector<vector<double>> a, vector<vector<double>> b);

vector<vector<double>> broadcast(vector<vector<double>> a, int col);

vector<vector<double>> Mat_Add(vector<vector<double>> a, vector<vector<double>> b);

vector<vector<double>> Mat_Sub(vector<vector<double>> a, vector<vector<double>> b);

vector<vector<double>> Mat_Div(vector<vector<double>> a, vector<vector<double>> b);

vector<vector<double>> Mult(vector<vector<double>> a, vector<vector<double>> b);

vector<vector<double>> transpose(vector<double> a);

vector<vector<double>> transpose(vector<vector<double>> a);

vector<vector<double>> max(vector<vector<double>> a, int axis);

vector<vector<double>> sum(vector<vector<double>> a, int axis);
