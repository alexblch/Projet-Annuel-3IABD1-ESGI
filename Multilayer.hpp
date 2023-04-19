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
    double *weight; // tableau de poids
    double bias;    // tableau de biais
    double *output; // tableau de sortie
    int data_size;
    int weight_size;
    int output_size;
    int *hidden_Layer; // tableau de couches cachées
    int hidden_Layer_size;
    MatrixXd image;
    MatrixXd weight_matrix;
    double sum;

public:
    Multilayer(int data_size, int weight_size, int bias, int output_size, MatrixXd image);
    double sigmoid(double x);
    double activation();
    double tanh(double x);
    double perceptron();
    void flatten(MatrixXd mat);
    void display_matrix();
    void setWeight();
    void displayWeight();
    void displaySum();
    void set_Data();
    void display_data();
    //~Multilayer();
};