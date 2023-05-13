#include "Multilayer.hpp"
#include <iostream>
extern "C"
{

    Multilayer::Multilayer(int data_size, int bias, int output_size, MatrixXd image) // constructor
    {
        output = new double[output_size];
        this->data_size = data_size;
        this->weight_size = weight_size;
        this->bias = bias;
        this->output_size = output_size;
        this->image = image;
    }

    void Multilayer::displaySum() { cout << "Sum : " << sum << endl; } // afficher la somme

    double Multilayer::sigmoid(double x) { return 1 / (1 + exp(-x)); }

    double Multilayer::tanh(double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); } // tanh function

    void Multilayer::display_matrix() { cout << image << endl; } // afficher la matrice

    void Multilayer::display_matrix_weight() { cout << weight_matrix << endl; } // afficher la matrice de poids

    void Multilayer::set_Data()
    {
        for (int i = 0; i < data.size(); i++)
            data[i] = data[i] / 255;
    } // setter

    void Multilayer::display_data() // afficher les données
    {
        cout << "Data : ";
        for (int i = 0; i < data.size(); i++)
            cout << data[i] << " ";
        cout << endl;
    }

    void Multilayer::displayWeight() // afficher les poids
    {
        cout << "Weight : ";
        for (int i = 0; i < weight_size; i++)
            cout << weight[i] << " ";
        cout << endl;
    }

    void Multilayer::flatten() // convertir une matrice en vecteur
    {
        vector<double> data;
        for (int i = 0; i < image.rows(); i++)
        {
            for (int j = 0; j < image.cols(); j++)
                data.push_back(image(i, j));
        }
        this->data = data;
    }

    void Multilayer::setWeight() // setter
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-6, 6);
        for (int i = 0; i < image.cols() * image.rows(); i++)
            weight.push_back((int)dis(gen));
    }

    double Multilayer::activation() // développement du perceptron
    {
        sum = 0;
        for (int i = 0; i < data.size(); i++)
        {
            sum += data[i] * weight[i];
            cout << sum << endl;
        }
        sum += bias;
        sum = tanh(sum);
        return sum;
    }

double Multilayer::perceptron() // fonction de sortie
{
    out = 0;
    sum = 0;
    int increment = 0;
    int data_increment = 0;
    //remplissage de la matrice de données
    for(int i = 0 ; i < data_matrix.rows(); i++)
    {
        sum += bias;
        while (data_increment < data.size() && increment < weight.size())
        {
            sum += data[data_increment] * weight[increment];
            data_increment++;
            increment++;
        }
        data_matrix(i,0) = tanh(sum);
        data_increment = 0;
        sum = 0;
    }
    for (int i = 0; i < data_matrix.rows(); i++)
    {
        for (int j = 1; j < data_matrix.cols(); j++)
        {
            data_matrix(i,j) = tanh(data_matrix(i,j-1));
        }
    }
    cout << data_matrix << endl;
    return out;
}

    void Multilayer::set_matrix_weight()
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-6, 6);
        MatrixXd matrix(data.size() * neurons, hidden_Layer-1);
        for (int i = 0; i < matrix.rows(); i++)
        {
            for (int j = 0; j < matrix.cols(); j++)
                matrix(i, j) = (int)dis(gen);
        }
        this->weight_matrix = matrix;
    }

    void Multilayer::set_hidden_layer()
    {
        int hiddenl, lay;
        cout << "Enter the number of hidden layers : ";
        cin >> hiddenl;
        cout << "Enter the number of neurons in each hidden layer : ";
        cin >> lay;
        this->hidden_Layer = hiddenl;
        this->neurons = lay;
    }

    void Multilayer::set_matrix_data()
    {
        MatrixXd current(neurons, hidden_Layer);
        for (int i = 0; i < current.rows(); i++)
        {
            for (int j = 0; j < current.cols(); j++)
                current(i, j) = 0;
        }
        this->data_matrix = current;
    }

    void Multilayer::display_dataMatrix()
    {
        cout << "Data matrix : " << endl;
        cout << data_matrix << endl;
    }
    /*double Multilayer::findOutput() // fonction de sortie
    {
        return activation(data, weight, bias, data_size);

    }*/

    /*Multilayer::~Multilayer(Multilayer *ml) // destructor
    {
        free(ml);
    }*/
}