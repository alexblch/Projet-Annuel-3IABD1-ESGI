#include "Multilayer.hpp"


Multilayer::Multilayer(int data_size, int weight_size, int bias, int output_size, MatrixXd image) // constructor
{
    weight = new double[weight_size];
    output = new double[output_size];
    this->data_size = data_size;
    this->weight_size = weight_size;
    this->bias = bias;
    this->output_size = output_size;
    this->image = image;
}




void Multilayer::displaySum() {  cout << "Sum : " << sum << endl;} // afficher la somme

double Multilayer::sigmoid(double x) {return 1 / (1 + exp(-x));}

double Multilayer::tanh(double x) {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));} // tanh function

void Multilayer::display_matrix() {cout << image << endl;} // afficher la matrice


void Multilayer::set_Data(){for(int i = 0; i < data.size(); i++)    data[i] = data[i] / 255;} // setter


void Multilayer::display_data() // afficher les données
{
    cout << "Data : ";
    for(int i = 0; i < data.size(); i++)
        cout << data[i] << " ";
    cout << endl;
}

void Multilayer::displayWeight() // afficher les poids
{
    for (int i = 0; i < weight_size; i++)
        cout << weight[i] << " ";
    cout << endl;
}




void Multilayer::flatten(MatrixXd mat) // convertir une matrice en vecteur
{
   vector <double> data;
    for (int i = 0; i < mat.rows() ; i++)
    {
        for (int j = 0; j < mat.cols(); j++)
            data.push_back((double)mat(i, j));
    }
    this->data = data;
}

void Multilayer::setWeight() // setter
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1, 5);
    for(int i = 0; i < image.cols() * image.rows(); i++)
        weight[i] = (int)dis(gen);
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

/*double Multilayer::findOutput() // fonction de sortie
{
    return activation(data, weight, bias, data_size);

}*/



/*Multilayer::~Multilayer(Multilayer *ml) // destructor
{
    free(ml);
}*/
