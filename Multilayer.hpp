#include <cmath>



class Multilayer
{
private:
    float *data; //tableau de données
    float *weight;//tableau de poids
    float *bias; //tableau de biais
    float *output; //tableau de sortie
    int data_size;
    int weight_size;
    int bias_size;
    int output_size;
    int *hidden_Layer; //tableau de couches cachées
    int hidden_Layer_size;
public:
    Multilayer(int data_size, int weight_size, int bias_size, int output_size);
    float sigmoid(float x);
    float *setBias (float n, float *bias, int bias_size);
    float activation(float *data, float *weight, float *bias, int data_size, int weight_size, int bias_size);
    ~Multilayer();
};

float Multilayer::sigmoid(float x) //sigmoid function
{
    return 1 / (1 + exp(-x));
}

float *Multilayer::setBias (float n, float *bias, int bias_size) //
{
    for (int i = 0; i < bias_size; i++)
        bias[i] = n;
    return bias;
}

float Multilayer::activation(float *data, float *weight, float *bias, int data_size, int weight_size, int bias_size) //développement du perceptron
{
    float sum = 0;
    for (int i = 0; i < data_size; i++)
        sum += data[i] * weight[i];
    for (int i = 0; i < bias_size; i++)
        sum += bias[i];
    return sigmoid(sum);
}

Multilayer::Multilayer(int data_size, int weight_size, int bias_size, int output_size) //constructor
{
    data = new float[data_size];
    weight = new float[weight_size];
    bias = new float[bias_size];
    output = new float[output_size];
    this->data_size = data_size;
    this->weight_size = weight_size;
    this->bias_size = bias_size;
    this->output_size = output_size;
}

Multilayer::~Multilayer()
{
}
