all:
	g++ -I /usr/include/eigen3 -o libadd.so -shared -fPIC function.cpp
	python3 main.py
	