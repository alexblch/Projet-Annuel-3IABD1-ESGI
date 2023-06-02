#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <random>
#include <vector>
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

extern "C"
{

    double perceptron(int hidden_Layer, int neurons, int random, double *data, int bias, int size) // fonction de sortie, perceptron multicouche
    {
        //display_tab(data, size);
        data = set_Data(data, size);
        //display_tab(data, size);
        // random number
        vector<double> weight;
        vector<double> weight_output;
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-random, random);
        for (int i = 0; i < neurons; i++)
            weight_output.push_back((int)dis(gen));
        //display_vector(weight_output);
        // cas NULL
        // remplir le vecteur de poids
        for (int i = 0; i < size * neurons; i++)
            weight.push_back((int)dis(gen));
        MatrixXd data_matrix(neurons, hidden_Layer);
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            for (int j = 0; j < data_matrix.cols(); j++)
                data_matrix(i, j) = 0;
        }
        //display_matrix(data_matrix, data_matrix.rows(), data_matrix.cols());
        MatrixXd weight_matrix(neurons * neurons, hidden_Layer - 1);
        // remplir le vecteur de poids de sortie
        for (int i = 0; i < neurons; i++)
            weight_output.push_back(dis(gen));
        // remplir la matrice de poids
        for (int i = 0; i < weight_matrix.rows(); i++)
        {
            for (int j = 0; j < weight_matrix.cols(); j++)
                weight_matrix(i, j) = dis(gen);
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
            //display_matrix(data_matrix, data_matrix.rows(), data_matrix.cols());
            //display_matrix(weight_matrix, weight_matrix.rows(), weight_matrix.cols());
        }
        double out = 0;
        out += bias;
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            out += data_matrix(i, data_matrix.cols() - 1) * weight_output[i];
            cout << out << endl;
        }
        //cout << "out : " << out << endl;
        out = tanh(out); // tanh
        //cout << "tanh : " << out << endl;
        return out;
    }

    double linear_model(double *data, double *weight, int size, int bias)
    {
        data = set_Data(data, size);
        //display_tab(data, size);
        double sum = 0;
        if (size == 0)
            return 404;
        sum += (double)bias;
        for (int i = 0; i < size; i++)
        {
            sum += data[i] * weight[i];
            cout << sum << " ";
        }
        cout << endl;
        /*cout << "sum : " << sum << endl;
        cout << "tanh : " << tanh(sum) << endl;*/
        sum = tanh(sum);
        if (sum < -0.33)
            return -1;
        if (sum > -0.33 && sum < 0.33)
            return 0;
        else
            return 1;
    }
}
