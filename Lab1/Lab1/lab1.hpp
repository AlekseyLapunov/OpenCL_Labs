#pragma once

#include <ocl.hpp>

class Lab1 {
public:
    Lab1(const std::string& kernelCodeFileName, const std::string& kernelFunctionName) {
        ocl::log::setLogStream(std::cout);
        ocl::log::enableColor();

        ocl::init(kernelCodeFileName, kernelFunctionName);
    }

    ~Lab1() {
        ocl::cleanup();
    }

    void doTask() {
        ocl::printVerboseInfo(1);

        const int arrSize = 5;                 // arg pos 1
        int arr[arrSize] = { 3, 5, 7, 9, 11 }; // arg pos 0

        ocl::addArgumentArray(arr, arrSize);   // arg pos 0
        ocl::addArgument(arrSize);             // arg pos 1

        std::cout << "arr[] before kernel calculations:\t";
        printArray(arr, arrSize);

        if (!ocl::promptExecuteKernel(0, arr))
            return;

        std::cout << "arr[] after kernel calculations:\t";
        printArray(arr, arrSize);
    }

private:
    void printArray(const int arr[], int size, std::ostream& ost = std::cout) const {
        for (int i = 0; i < size; i++) 
            ost << arr[i] << " ";
        ost << "\n";
    }

private:
    Lab1(const Lab1& other) {}
    Lab1& operator=(const Lab1& other) {}
    Lab1(Lab1&& other) noexcept {}
    Lab1& operator=(const Lab1&& other) noexcept {}
};
