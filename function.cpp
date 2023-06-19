#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cstdio>
#include <fstream>
using namespace std;
using namespace Eigen;

double *set_Data(double *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = data[i] / 255;
    }
    cout << endl;
    return data;
}

// g++ -fPIC -shared -I /usr/include/eigen3 function.cpp -o libadd.so
void display_matrix(MatrixXd mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            cout << mat(i, j) << " ";
        cout << endl;
    }
    cout << endl;
}

void display_vector(vector<double> vec)
{
    cout << "vector : " << endl;
    for (int i = 0; i < vec.size(); i++)
        cout << vec[i] << " ";
    cout << endl
         << endl;
}

void display_tab(double *tab, int size)
{
    cout << "tab : " << endl;
    for (int i = 0; i < size; i++)
        cout << tab[i] << " ";
    cout << endl
         << endl;
}

void matrixinfile(ofstream &file, MatrixXd mat)
{
    for (int i = 0; i < mat.rows(); i++)
    {
        for (int j = 0; j < mat.cols(); j++)
            file << mat(i, j) << " ";
        file << endl;
    }
    file << endl;
}

void set_matrix(MatrixXd &mat, int random)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-random - 1, random + 1);
    for (int i = 0; i < mat.rows(); i++)
    {
        for (int j = 0; j < mat.cols(); j++)
            mat(i, j) = int(dis(gen));
    }
}

extern "C"
{
    double perceptron(int hidden_Layer, int neurons, int random, double *data, int bias, int size) // fonction de sortie, perceptron multicouche
    {
        ofstream file("weight.txt");
        ofstream matrixfile("data_matrix.txt");
        ofstream weightfile("weight_matrix.txt");
        data = set_Data(data, size);

        vector<double> weight;
        vector<double> weight_output;
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-random - 1, random + 1);

        int number = 0;

        for (int i = 0; i < size * neurons; i++)
        {
            if ((size * neurons) / 2 > i)
                weight.push_back(1);
            else
                weight.push_back(-1);
        }
        for (int i = 0; i < neurons; i++)
            weight_output.push_back(int(dis(gen) * 100));
        if (file.is_open())
        {
            file << "weight_output :" << endl;
            for (int i = 0; i < weight_output.size(); i++)
            {
                file << weight_output[i] / 100 << " ";
            }
            file << endl;
        }
        MatrixXd data_matrix = MatrixXd::Zero(neurons, hidden_Layer);
        // display_vector(weight);
        // display_matrix(data_matrix, data_matrix.rows(), data_matrix.cols());
        MatrixXd weight_matrix = MatrixXd::Zero(neurons * hidden_Layer, hidden_Layer - 1);
        // remplir le vecteur de poids de sortie
        // remplir la matrice de poids
        set_matrix(weight_matrix, random);
        if (weightfile.is_open())
        {
            matrixinfile(weightfile, weight_matrix);
        }
        int sum = 0;
        if (hidden_Layer == 0 || neurons == 0)
        {
            return 404;
        }
        // sinon remplissage des couches cachées
        sum = 0;
        int increment = 0;
        int data_increment = 0;
        int bias2 = -(bias);
        // remplissage de la matrice de données
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            if (i % 2 == 0)
                sum += bias;
            else
                sum += bias2;
            while (data_increment < size)
            {
                sum += data[data_increment] * weight[increment];
                data_increment++;
                increment++;
            }
            data_matrix(i, 0) = tanh(sum);
            data_increment = 0;
            sum = 0;
        }
        // display_dataMatrix();
        // remplissage des autres couches
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
                data_matrix(j, i) = tanh(sum);
                data_increment = 0;
                sum = 0;
            }
        }
        double out = 0;
        out += bias;
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            out += data_matrix(i, data_matrix.cols() - 1) * weight_output[i];
        }
        if(matrixfile.is_open())
        {
            matrixinfile(matrixfile, data_matrix);
        }
        double error = 0;

        return tanh(out);
    }

    double linear_model(double *data, double *weight, int size, int bias)
    {
        data = set_Data(data, size);
        double sum = 0;
        if (size == 0)
            return 404;
        sum += (double)bias;
        for (int i = 0; i < size; i++)
        {
            sum += data[i] * weight[i];
        }
        sum = tanh(sum);
        if (sum < -0.33)
            return -1;
        if (sum > -0.33 && sum < 0.33)
            return 0;
        else
            return 1;
    }
}
