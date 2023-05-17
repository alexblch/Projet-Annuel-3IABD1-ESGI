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

int *flatten(int **mat, int rows, int cols)
{
    int *array = new int[rows * cols];
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
