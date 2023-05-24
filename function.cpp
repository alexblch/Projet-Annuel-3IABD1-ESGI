#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <random>
#include <vector>
using namespace std;
using namespace Eigen;


extern "C"
{
    double tanh(double x)
    {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    double *flatten(MatrixXd mat, int rows, int cols)
    {
        double *array = new double[rows * cols];
        int k = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[k] = mat(i,j);
                k++;
            }
        }
        return array;
    }

    double *set_Data(double *data, int size)
    {
        for(int i = 0; i < size; i++)
            data[i] = data[i] / 255;
        return data;
    }

    void display_matrix(int **mat, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
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
        if(tanh(sum) > 0.5)
            return 1;
        if(tanh(sum) < -0.5)
            return -1;
        else
            return 0;
    }

    void set_Hiddenlayer(int &neuron, int &hidden_layer)
    {
        cout << "Enter the number of hidden layer : ";
        cin >> hidden_layer;
        cout << "Enter the number of neurons in each hidden layer : ";
        cin >> neuron;
    }


    
    double* set_weight_output(int neurons, int random)
    {
        double* weight_output = new double[neurons];
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-(random), (random));
        for (int i = 0; i < neurons; i++)
            weight_output[i] = (int)dis(gen);
        return weight_output;
    }

}

//g++ -fPIC -shared -I /usr/include/eigen3 function.cpp -o libadd.so
