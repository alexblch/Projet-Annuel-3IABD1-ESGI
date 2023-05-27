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
        {
            data[i] = data[i] / 255;
            cout << data[i] << " ";
        }
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

    double perceptron(int hidden_Layer, int neurons, int random, double *data, int bias, int size) // fonction de sortie, perceptron multicouche
    {
        //random number
        vector<double> weight;
        vector <double> weight_output;
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-random, random);
        // cas NULL
        //remplir le vecteur de poids
        for (int i = 0; i < size * neurons; i++)
        {
            weight.push_back(dis(gen));
        }
        
        for (int i = 0; i < size * neurons; i++)
        {
            weight[i] = dis(gen);
        }
        MatrixXd data_matrix(neurons, hidden_Layer);
        MatrixXd weight_matrix(neurons*neurons, hidden_Layer-1);
        int sum = 0;
        if(hidden_Layer == 0 || neurons == 0)
        {
            int out = 0;
            sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += data[i] * weight[i]; 
            }
            sum += bias;
            out = tanh(sum);
            return out;
        } 
        //sinon remplissage des couches cachées
        int out = 0;
        sum = 0;
        int increment = 0;
        int data_increment = 0;
        // remplissage de la matrice de données
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            sum += bias;
            while (data_increment < size && increment < weight.size())
            {
                sum += data[data_increment] * weight[increment];
                data_increment++;
                increment++;
            }
            data_matrix(i, 0) = tanh(sum);
            data_increment = 0;
            sum = 0;
        }
        //display_dataMatrix();
        //remplissage des autres couches
        for (int i = 1; i < data_matrix.cols(); i++)
        {
            for (int j = 0; j < data_matrix.rows(); j++)
            {
                sum += bias;
                while (data_increment < data_matrix.rows() && increment < weight_matrix.rows())
                {
                    sum += data_matrix(data_increment, i - 1) * weight_matrix(increment, i - 1);
                    data_increment++;
                    increment++;
                }
                data_matrix(j,i) = (exp(sum) - exp(-sum)) / (exp(sum) + exp(-sum));
                data_increment = 0;
                sum = 0;
            }
            //display_dataMatrix();
        }
        out = 0;
        out += bias;
        for (int i = 0; i < data_matrix.rows(); i++)
            out += data_matrix(i, data_matrix.cols() - 1) * weight_output[i];
        out = (exp(out) - exp(-out)) / (exp(out) + exp(-out)); // tanh
        return out;
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
        sum += (double)bias;
        for (int i = 0; i < size; i++)
        {
            sum += data[i] * weight[i];
            cout << sum << " " << endl;
        }
        return sum;
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
