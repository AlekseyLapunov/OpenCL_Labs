#include <iostream>

#include "lab3.hpp"

int main(int argc, char* argv[]) {
    Lab3 lab3("__kernel_lab3_task.c", "solve");

    lab3.generateFile("input.txt", 2, 3, 1000000);
    lab3.importFromFile("input.txt");

    lab3.cpuSolve();
    lab3.gpuSolve(256);
    lab3.printResults();

    lab3.exportToFile("output.txt");
    return 0;
}
