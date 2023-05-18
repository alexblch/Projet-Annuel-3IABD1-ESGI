#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

double tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double *flatten(int **mat, int rows, int cols)
{
    double *array = new double[rows * cols];
    int k = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols;j++)
        {
            array[k] = mat[i][j];
            k++;
        }
    }
    return array;
}

double** set_Image(double **mat, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols;j++)
            mat[i][j] /= 255;
    }
    return mat;
}

void display_matrix(int **mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols;j++)
            cout << mat[i][j] << " ";
        cout << endl;
    }
}

void display_array(int *array, int size)
{
    for (int i = 0; i < size; i++)
        cout << array[i] << " ";
    cout << endl;
}

double linear_model(double *data, double *weight, int size, int bias)
{
    double sum = 0;
    sum += bias;
    for (int i = 0; i < size; i++)
        sum += data[i] * weight[i];
    return sum;
}


void set_Hiddenlayer(int &neuron, int &hidden_layer)
{
    cout << "Enter the number of hidden layer : ";
    cin >> hidden_layer;
    cout << "Enter the number of neurons in each hidden layer : ";
    cin >> neuron;
}

