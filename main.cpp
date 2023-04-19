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
    //random number
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 255);
    MatrixXd image(4,4);
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
            image(i,j) = dis(gen);
    }
    int weight = image.cols() * image.rows();
    Multilayer *ml = new Multilayer(4,weight,2,2, image);
    ml->flatten(image);
    ml->setWeight();
    ml->displayWeight();
    ml->set_Data();
    ml->display_data();
    ml->display_matrix();
    ml->activation();
    ml->displaySum();
    return 0;
}
// compile with eigen3 and c++11
// g++ -std=c++11 main.cpp Multilayer.cpp -I /usr/include/eigen3 -o main