#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "Multilayer.hpp"

using namespace std;
using namespace Eigen;

int main()
{
    int rows, cols;
    cout << "Welcome to the ML maker, enter number of rows and columns : (%d %d)" << endl;
    cin >> rows >> cols;
    //random number
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 256);
    MatrixXd image(rows,cols);
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
            image(i,j) = (int)dis(gen);
    }
    int weight = image.cols() * image.rows();
    Multilayer *ml = new Multilayer(4,weight,2,2, image);
    ml->flatten();
    ml->setWeight();
    ml->displayWeight();
    ml->set_Data();
    ml->display_data();
    ml->display_matrix();
    ml->activation();
    ml->displaySum();
    ml->set_hidden_layer();
    ml->set_matrix_weight();
    ml->display_matrix_weight();
    ml->set_matrix_data();
    ml->display_dataMatrix();
    return 0;
}
// compile with eigen3 and c++11
//g++ -std=c++11 main.cpp Multilayer.cpp -I /usr/include/eigen3 -o prog