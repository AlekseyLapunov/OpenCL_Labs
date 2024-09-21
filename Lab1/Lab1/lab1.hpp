#pragma once

#include <ocl.hpp>

class Lab1 {
public:
    Lab1(const std::string& kernelCodeFileName, const std::string& kernelFunctionName) {
        ocl::init(kernelCodeFileName, kernelFunctionName);
        ocl::log::setLogStream(std::cout);
    }
    ~Lab1() {
        ocl::cleanup();
    }

    void doTask() {
        ocl::printVerboseInfo(1);

        const int arrSize = 5;                 // arg 1
        int arr[arrSize] = { 3, 5, 7, 9, 11 }; // arg 0

        ocl::kernel::emplaceArgumentArray((void*)&arr, sizeof(int), arrSize);
        ocl::kernel::emplaceArgument((void*)&arrSize, sizeof(int));

        std::cout << "arr[] before kernel calculations:\t";
        printArray(arr, arrSize);

        size_t globalWorkSize[] = { 4, 0, 0 };
        size_t localWorkSize[] = { 0, 0, 0 };

        if (!ocl::executeKernel(1, globalWorkSize, true, localWorkSize, 0, (void*)&arr))
            return;

        std::cout << "arr[] after kernel calculations:\t";
        printArray(arr, arrSize);
    }

private:
    void printArray(int arr[], int size, std::ostream& ost = std::cout) const {
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
