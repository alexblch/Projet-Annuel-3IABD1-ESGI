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
    Multilayer *ml = new Multilayer(4,4,2,2);
    return 0;
}
// compile with eigen3 and c++11
// g++ -std=c++11 main.cpp Multilayer.cpp -I /usr/include/eigen3 -o main