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

void getMatrixfromfile(ifstream &file, MatrixXd &mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
            file >> mat(i, j);
    }
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

void vectorinfile(ofstream &file, vector<double> vec)
{
    for (int i = 0; i < vec.size(); i++)
        file << vec[i] << " ";
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

void file_data(double *data, int size, ofstream &file)
{
    for (int i = 0; i < size; i++)
        file << data[i] << " ";
    file << endl;
}

void get_data_infile(ifstream &file, double *data, int size)
{
    for (int i = 0; i < size; i++)
        file >> data[i];
}

void get_vector_infile(ifstream &file, vector<double> &vec, int size)
{
    double value;
    for (int i = 0; i < size; i++)
    {
        file >> value;
        vec.push_back(value);
    }
}

void set_vector_weight(vector<double> &weight, int size, int neurons, int random)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-random - 1, random + 1);
    for (int i = 0; i < size * neurons; i++)
    {
        if (i == 0 || i == size * neurons - 1)
            weight.push_back(1);
        else
        {
            if (neurons < 3)
                weight.push_back(0);
            else
                weight.push_back(int(dis(gen)));
        }
    }
}


void stochastic_gradient_descent(int hidden_Layer, int neurons, MatrixXd data_matrix, MatrixXd weight_matrix, vector<double> weight, vector<double> weight_output, double out)
{
    double delta = 1 - out;
    return;
}

extern "C"
{

    void create_file(int hidden_layers, int neurons, int random)
    {
        ofstream file("file/weight.txt");
        ofstream weightfile("file/weight_matrix.txt");
        ofstream weight_outputfile("file/weight_output.txt");
        //random number
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-random, random);
        MatrixXd weight_matrix = MatrixXd::Zero(neurons * hidden_layers, hidden_layers - 1);
        std::vector<double> weight;
        std::vector<double> weight_output;
        for (int i = 0; i < neurons * hidden_layers; i++)
            weight.push_back(int(dis(gen)));
        if(file.is_open())
            vectorinfile(file, weight);
        for (int i = 0; i < neurons; i++)
            weight_output.push_back(int(dis(gen)));
        if(weight_outputfile.is_open())
            vectorinfile(weight_outputfile, weight_output);
        for (int i = 0; i < weight_matrix.rows(); i++)
        {
            for (int j = 0; j < weight_matrix.cols(); j++)
                weight_matrix(i, j) = int(dis(gen));
        }
        //stock in file
        if (weightfile.is_open())
        {
            matrixinfile(weightfile, weight_matrix);
        }
    }


    double perceptron(int hidden_Layer, int neurons, int random, double *data, int bias, int size) // fonction de sortie, perceptron multicouche
    {
        ifstream file("file/weight.txt");
        ifstream weightfile("file/weight_matrix.txt");
        ifstream weight_outputfile("file/weight_output.txt");
        ifstream weight_matrixfile("file/weight_matrix.txt");

        MatrixXd data_matrix = MatrixXd::Zero(neurons, hidden_Layer);
        MatrixXd weight_matrix = MatrixXd::Zero(neurons * hidden_Layer, hidden_Layer - 1);
        data = set_Data(data, size);
        if (weight_matrixfile.is_open())
        {
            getMatrixfromfile(weight_matrixfile, weight_matrix, weight_matrix.rows(), weight_matrix.cols());
        }
        
        vector<double> weight;
        if(file.is_open())
            get_vector_infile(file, weight, neurons);

        vector<double> weight_output;
        if (weight_outputfile.is_open())
            get_vector_infile(weight_outputfile, weight_output, neurons);
        

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-random , random );

        int number = 0;


        // remplir le vecteur de poids de sortie
        // remplir la matrice de poids
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
