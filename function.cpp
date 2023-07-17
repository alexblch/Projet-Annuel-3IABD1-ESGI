#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cstdio>
#include <fstream>
#include <cmath>
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
        for (int j = 0; j < cols ; j++)
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
    cout << "taille : " << vec.size() << endl
         << endl;
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
    file.close();
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



void retropropagation(int hidden_Layer, int neurons, double *data, MatrixXd &data_matrix, MatrixXd &weight_matrix, vector<double> &weight, vector<double> &weight_output, double *out, int nbClass, int *prediction, int size, double learning_rate/*double bias*/ )
{
    double *delta = new double[nbClass];
    for(int i = 0; i < nbClass; i++)
        delta[i] = prediction[i] - out[i];
    
    std::ofstream loss("file/loss.txt");
    if(loss)
    {
        file_data(delta, nbClass, loss);
    }

    MatrixXd delta_matrix = MatrixXd::Zero(neurons, hidden_Layer);
    double sum = 0;

    for(int i = 0 ; i < neurons; i++)
    {
        sum = 0;
        for (int k = 0; k < nbClass; k++)
            sum += delta[k] * weight_output[k * neurons + i];

        delta_matrix(i, hidden_Layer-1) = (1 - data_matrix(i, hidden_Layer-1) * data_matrix(i, hidden_Layer-1)) * sum;
    }

    for (int i = hidden_Layer-2; i >= 0 ; i--)
    {
        for (int j = 0; j < neurons; j++)
        {
            sum = 0;
            for(int k = 0; k < neurons; k++)
            {
                sum += delta_matrix(k, i+1) * weight_matrix(k * neurons + j, i);
            }
            delta_matrix(j, i) = (1 - data_matrix(j, i) * data_matrix(j, i)) * sum;
        }
    }

    ofstream deltas("file/delta.txt");
    matrixinfile(deltas, delta_matrix);

    // stochastic gradient descent
    for(int i = 0 ; i < neurons ; i++)
    {
        for(int j = 0 ; j < size ; j++)
        {
            weight[i * size + j] -= learning_rate * data[j] * delta_matrix(i, 0);
        }
    }

    for (int i = 0; i < neurons; i++)
    {
        for (int j = 0; j < hidden_Layer - 1; j++)
        {
            weight_matrix(i * neurons + j, j) -= learning_rate * data_matrix(i, j) * delta_matrix(i, j+1);
        }
    }

    for(int i = 0 ; i < nbClass ; i++)
    {
        for(int j = 0 ; j < neurons ; j++)
        {
            weight_output[i * neurons + j] -= learning_rate * data_matrix(j, hidden_Layer-1) * delta[i];
        }
    }

    std::ofstream weightfile("file/weight.txt");
    vectorinfile(weightfile, weight);

    std::ofstream weight_matrixfile("file/weight_matrix.txt");
    matrixinfile(weight_matrixfile, weight_matrix);

    std::ofstream weight_outputfile("file/weight_output.txt");
    vectorinfile(weight_outputfile, weight_output);
}

extern "C"
{
    // stock datas in file, facilitate for stochastic descent gradient
    void create_file(int hidden_layers, int neurons, int random, int size_image, int nbClass)
    {
        ofstream file("file/weight.txt");
        ofstream weightfile("file/weight_matrix.txt");
        ofstream weight_outputfile("file/weight_output.txt");
        // random number
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-random, random);
        MatrixXd weight_matrix = MatrixXd::Zero(neurons * neurons, hidden_layers - 1);
        std::vector<double> weight;
        std::vector<double> weight_output;
        
        for (int i = 0; i < neurons * size_image; i++)
        {
            weight.push_back(float(dis(gen)));
        }
        if (file.is_open())
            vectorinfile(file, weight);
        for (int i = 0; i < neurons * nbClass; i++)
        {
            weight_output.push_back(float(dis(gen)));
        }
        if (weight_outputfile.is_open())
            vectorinfile(weight_outputfile, weight_output);
        for (int i = 0; i < weight_matrix.rows(); i++)
        {
            for (int j = 0; j < weight_matrix.cols(); j++)
                weight_matrix(i, j) = float(dis(gen));
        }
        // stock in file
        if (weightfile.is_open())
        {
            matrixinfile(weightfile, weight_matrix);
        }
        weightfile.close();
        weight_outputfile.close();
        file.close();
    }

    double perceptron(int hidden_Layer, int neurons, int random, double *data, double bias, int size, int nbClass, int *prediction, double learning_rate) // fonction de sortie, perceptron multicouche
    {
        ifstream file("file/weight.txt");
        ifstream weightfile("file/weight_matrix.txt");
        ifstream weight_outputfile("file/weight_output.txt");
        ifstream weight_matrixfile("file/weight_matrix.txt");

        ofstream file_data_image("file/data.txt");

        MatrixXd data_matrix(neurons, hidden_Layer);
        //aficher type de data_matrix
        data_matrix.setConstant(bias);
        MatrixXd weight_matrix = MatrixXd::Zero(neurons * neurons, hidden_Layer - 1);

        data = set_Data(data, size);

        if (file_data_image.is_open())
            file_data(data, size, file_data_image);
        file_data_image.close();

        if (weight_matrixfile.is_open())
            getMatrixfromfile(weight_matrixfile, weight_matrix, weight_matrix.rows(), weight_matrix.cols());
        weight_matrixfile.close();
        cout << endl;
        vector<double> weight;
        if (file.is_open())
            get_vector_infile(file, weight, neurons*size);
        file.close();

        vector<double> weight_output;
        if (weight_outputfile.is_open())
            get_vector_infile(weight_outputfile, weight_output, neurons*nbClass);
        weight_outputfile.close();

        double sum = 0;
        if (hidden_Layer == 0 || neurons == 0)
        {
            std::runtime_error("hidden_Layer or neurons is null");
            return 0;
        }
        // sinon remplissage des couches cachées
        int increment = 0;



        // debut de la propagation
        //  remplissage de la matrice de données
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            //data_matrix(i, 0) += bias;
            for (int j = 0; j < size; j++)
            {
                data_matrix(i, 0) += data[j] * weight[increment];
                increment++;
            }
        }
        // Appliquer la fonction d'activation à la sortie de chaque neurone
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            data_matrix(i, 0) = std::tanh(data_matrix(i, 0));
        }
        increment = 0;


        // display_dataMatrix();
        // remplissage des autres couches
        for (int i = 1; i < data_matrix.cols(); i++)
        {
            for (int j = 0; j < data_matrix.rows(); j++)
            {
                //data_matrix(j, i) += bias;
                for (int k = 0; k < data_matrix.rows(); k++)
                {
                    data_matrix(j, i) += data_matrix(k, i - 1) * weight_matrix(increment, i - 1);
                    increment++;
                }
                data_matrix(j, i) = std::tanh(data_matrix(j, i));
                increment = 0;
            }
        }
        
        double *out = new double[nbClass];
        for (int i = 0; i < nbClass; i++)
        {
            out[i] = bias;
        }

        increment = 0;
        for (int i = 0; i < nbClass; i++)
        {
            for (int j = 0; j < data_matrix.rows(); j++)
            {
                out[i] += data_matrix(j, data_matrix.cols() - 1) * weight_output[increment];
                increment++;
            }
            out[i] = std::tanh(out[i]);
        }


        std::ofstream out_class("file/outclass.txt");
        if (out_class.is_open())
            file_data(out, nbClass, out_class);

        ofstream datafile("file/data_matrix.txt");
        if (datafile.is_open())
            matrixinfile(datafile, data_matrix);
        datafile.close();

        ofstream outfile("file/output.txt");
        retropropagation(hidden_Layer, neurons, data, data_matrix, weight_matrix, weight, weight_output, out, nbClass, prediction, size, learning_rate);
        return out[0];
    }

    double propagate(int hidden_Layer, int neurons, int random, double *data, double bias, int size, int nbClass)
    {
        ifstream file("file/weight.txt");
        ifstream weightfile("file/weight_matrix.txt");
        ifstream weight_outputfile("file/weight_output.txt");
        ifstream weight_matrixfile("file/weight_matrix.txt");

        ofstream file_data_image("file/data.txt");

        MatrixXd data_matrix(neurons, hidden_Layer);
        //aficher type de data_matrix
        data_matrix.setConstant(bias);
        MatrixXd weight_matrix = MatrixXd::Zero(neurons * neurons, hidden_Layer - 1);

        data = set_Data(data, size);

        if (file_data_image.is_open())
            file_data(data, size, file_data_image);
        file_data_image.close();

        if (weight_matrixfile.is_open())
            getMatrixfromfile(weight_matrixfile, weight_matrix, weight_matrix.rows(), weight_matrix.cols());
        weight_matrixfile.close();
        cout << endl;
        vector<double> weight;
        if (file.is_open())
            get_vector_infile(file, weight, neurons*size);
        file.close();

        vector<double> weight_output;
        if (weight_outputfile.is_open())
            get_vector_infile(weight_outputfile, weight_output, neurons*nbClass);
        weight_outputfile.close();

        double sum = 0;
        if (hidden_Layer == 0 || neurons == 0)
        {
            std::runtime_error("hidden_Layer or neurons is null");
            return 0;
        }
        // sinon remplissage des couches cachées
        int increment = 0;



        // debut de la propagation
        //  remplissage de la matrice de données
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            //data_matrix(i, 0) += bias;
            for (int j = 0; j < size; j++)
            {
                data_matrix(i, 0) += data[j] * weight[increment];
                increment++;
            }
        }
        // Appliquer la fonction d'activation à la sortie de chaque neurone
        for (int i = 0; i < data_matrix.rows(); i++)
        {
            data_matrix(i, 0) = std::tanh(data_matrix(i, 0));
        }
        increment = 0;


        // display_dataMatrix();
        // remplissage des autres couches
        for (int i = 1; i < data_matrix.cols(); i++)
        {
            for (int j = 0; j < data_matrix.rows(); j++)
            {
                //data_matrix(j, i) += bias;
                for (int k = 0; k < data_matrix.rows(); k++)
                {
                    data_matrix(j, i) += data_matrix(k, i - 1) * weight_matrix(increment, i - 1);
                    increment++;
                }
                data_matrix(j, i) = std::tanh(data_matrix(j, i));
                increment = 0;
            }
        }
        
        double *out = new double[nbClass];
        for (int i = 0; i < nbClass; i++)
        {
            out[i] = bias;
        }

        increment = 0;
        for (int i = 0; i < nbClass; i++)
        {
            for (int j = 0; j < data_matrix.rows(); j++)
            {
                out[i] += data_matrix(j, data_matrix.cols() - 1) * weight_output[increment];
                increment++;
            }
            out[i] = std::tanh(out[i]);
        }


        std::ofstream out_class("file/outclass.txt");
        if (out_class.is_open())
            file_data(out, nbClass, out_class);

        ofstream datafile("file/data_matrix.txt");
        if (datafile.is_open())
            matrixinfile(datafile, data_matrix);
        datafile.close();
        return out[0];
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
