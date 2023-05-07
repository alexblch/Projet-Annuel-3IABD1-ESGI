#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <random>

using namespace Eigen;
using namespace std;

class Multilayer
{
private:
    vector <double>data;   // tableau de données
    vector <float>weight; // tableau de poids
    double bias;    // tableau de biais
    double *output; // tableau de sortie
    int data_size;
    int weight_size;
    int output_size;
    int hidden_Layer; // nombre de couches cachées
    MatrixXd image; // matrice d'image
    MatrixXd weight_matrix; // matrice de poids du réseau
    MatrixXd data_matrix; // matrice de données du réseau
    int neurons; // nombre de neurones dans la couche cachée
    double sum;
    double out;

public:
    Multilayer(int data_size, int bias, int output_size, MatrixXd image);
    double sigmoid(double x);
    double activation();
    double tanh(double x);
    double perceptron();
    void flatten();
    void display_matrix();
    void setWeight();
    void displayWeight();
    void displaySum();
    void set_Data();
    void display_data();
    void set_matrix_weight();
    void display_matrix_weight();
    void set_matrix_data();
    void set_hidden_layer();
    void display_dataMatrix();
    //~Multilayer();
};