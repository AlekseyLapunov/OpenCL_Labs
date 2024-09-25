#include <iostream>

#include <ocl.hpp>

#include "lab2.hpp"

int main(int argc, char* argv[]) {
    Lab2 lab2("__kernel_lab2_task.c", "solve");

    lab2.generateFile("input.txt", 4, 5, 100000);
    lab2.importFromFile("input.txt");

    lab2.cpuSolve();
    lab2.gpuSolve(768);
    lab2.printResults();

    lab2.exportToFile("output.txt");
    return 0;
}
