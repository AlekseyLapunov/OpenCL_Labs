#include <iostream>

#include <ocl.hpp>

#include "lab1.hpp"

int main(int argc, char* argv[]) {
    Lab1 lab("__kernel_array.c", "SimpleWithArray");
    lab.doTask();
    return 0;
}
