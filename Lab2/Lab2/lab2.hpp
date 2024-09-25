#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

class Lab2 {
private:
    struct Problem {
        int N = 0, K = 0;
        size_t phoneNumbersCount = 0;
        std::vector<std::string> phoneNumbers;
    } _problem;

    struct Timer {
        Timer() : duration({}) {
            start = std::chrono::high_resolution_clock::now(); 
        }

        double getMs() {
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            return duration.count();
        }

        std::chrono::time_point<std::chrono::steady_clock> start, end;
        std::chrono::duration<double, std::milli> duration;
    };

    double cpuSolveTimeMs = 0.0f;
    double gpuSolveTimeMs = 0.0f;

    char* numbersBuffer = nullptr;
    std::vector<std::string> gpuResults;
    std::unordered_set<std::string> validationSet;

public:
    Lab2(const std::string& kernelCodeFileName, const std::string& kernelFunctionName) {
        ocl::log::setLogStream(std::cout);
        ocl::log::enableColor();

        ocl::init(kernelCodeFileName, kernelFunctionName);
    }

    ~Lab2() {
        ocl::cleanup();

        if (numbersBuffer != nullptr)
            delete[] numbersBuffer;
    }

    void gpuSolve(size_t workers) {
        if (numbersBuffer != nullptr)
            delete[] numbersBuffer;

        std::ostringstream ss;
        numbersBuffer = new char[_problem.phoneNumbersCount*13];

        for (size_t i = 0; i < _problem.phoneNumbersCount; i++)
            ss << _problem.phoneNumbers.at(i);

        strcpy_s(numbersBuffer, _problem.phoneNumbersCount*13, ss.str().c_str());

        ocl::addArgument(_problem.N);
        ocl::addArgument(_problem.K);
        ocl::addArgument(_problem.phoneNumbersCount);
        ocl::addArgumentArray(*numbersBuffer, _problem.phoneNumbersCount*12);

        Timer t;

        size_t gWorkSize[3] = { workers };
        ocl::executeKernel(1, gWorkSize, true, nullptr, 3, *numbersBuffer);
        gpuSolveTimeMs = t.getMs();

        std::cout << __FUNCTION__ << " elapsed time: " << gpuSolveTimeMs << "ms \n";

        bufferToResults();
    }

    void cpuSolve() {
        validationSet.clear();
        Timer t;

        char kDigit = _problem.K + '0';
        for (const auto& number : _problem.phoneNumbers) {
            int count = 0;
            for (const auto& digit : number) {
                if (digit == kDigit)
                    count++;
            }
            if (count == _problem.N)
                validationSet.insert(number);
        }

        cpuSolveTimeMs = t.getMs();
        std::cout << __FUNCTION__ << " elapsed time: " << cpuSolveTimeMs << " ms \n";
    }

    void importFromFile(const std::string& fileName) {
        std::ifstream ifst(fileName);

        if (!ifst.is_open()) {
            std::cout << "Can't open input file\n";
            return;
        }
        ifst.clear();
        ifst.seekg(0, std::ios::beg);

        ifst >> _problem.N;
        ifst >> _problem.K;
        ifst >> _problem.phoneNumbersCount;
        _problem.phoneNumbers.clear();
        _problem.phoneNumbers.resize(_problem.phoneNumbersCount);

        std::string line;
        std::getline(ifst, line);
        for (size_t i = 0; i < _problem.phoneNumbersCount; i++) {
            std::getline(ifst, line);
            _problem.phoneNumbers[i] = line;
        }

        std::cout << "Imported values from \"" << fileName << "\":\n";
        std::cout << "\tN = " << _problem.N << ", K = " << _problem.K;
        std::cout << ", Phone numbers count = " << _problem.phoneNumbers.size() << ".\n";

        ifst.close();
    }

    void exportToFile(const std::string& fileName) {
        std::ofstream ofst(fileName);

        if (!ofst.is_open()) {
            std::cout << "Can't open output file\n";
            return;
        }
        ofst.clear();
        ofst.seekp(0, std::ios::beg);

        ofst << gpuResults.size() << " numbers with exactly " << _problem.N << " (N) digits of " << _problem.K << " (K)\n";
        for (const auto& res : gpuResults)
            ofst << res << "\n";

        ofst.close();
        std::cout << "Results file \"" << fileName << "\" successfully written.\n";
    }

    void generateFile(const std::string& fileName, int N, int K, size_t phoneNumbers) {
        std::ofstream ofst(fileName);

        if (!ofst.is_open()) {
            std::cout << "Can't open output file\n";
            return;
        }
        ofst.clear();
        ofst.seekp(0, std::ios::beg);

        ofst << N << ' ' << K << ' ' << phoneNumbers << '\n';

        std::unordered_set<std::string> phoneNumbersSet;
        phoneNumbersSet.reserve(phoneNumbers);

        std::srand(std::time(nullptr));
        for (size_t i = 0; i < phoneNumbers; i++) {
            std::ostringstream ss;
            ss << "+7";
            for (uint8_t j = 0; j < 10; j++)
                ss << (rand() % 10);
            ss << '\n';
            
            if (!phoneNumbersSet.insert(ss.str()).second)
                i--;
        }

        for (const auto& number : phoneNumbersSet)
            ofst << number;

        std::cout << "File \"" << fileName << "\" generated with " << phoneNumbersSet.size() << " phone numbers.";
        std::cout << " N = " << N << ", K = " << K << ".\n";

        ofst.close();
    }

    void printResults() {
        size_t errorsCount = 0;
        for (const auto& res : gpuResults) {
            if (validationSet.find(res) == validationSet.end())
                errorsCount++;
        }
        errorsCount += validationSet.size() >= gpuResults.size() ? validationSet.size() - gpuResults.size() : gpuResults.size() - validationSet.size();
        float equalityPercent = static_cast<float>((validationSet.size() - errorsCount)/validationSet.size())*100.0f;

        std::ostringstream res;
        const int fillerSize = 50;
        res.setf(std::ios::fixed);
        res.precision(2);
        res << '\n';
        res << ocl::utils::fillerWithFileName("Results", fillerSize, ':');
        res << "1. Printing validation results:\n";
        res << ocl::utils::filler(fillerSize, '=');
        res << "CPU results count: " << validationSet.size() << "\n";
        res << "GPU results count: " << gpuResults.size() << "\n";
        res << ocl::utils::filler(fillerSize, '-');
        res << "Results equality percent: " << equalityPercent << "% ";
        res << ((equalityPercent == 100.0f) ? OCL_MAKE_GREEN("RESULTS VALID") : OCL_MAKE_RED("RESULTS NOT VALID"));
        res << '\n';
        res << ocl::utils::filler(fillerSize, '=');
        res << '\n';

        res << "2. Printing time results:\n";
        res.precision(3);
        res << ocl::utils::filler(fillerSize, '=');
        res << "Overall count of phone numbers:\t" << _problem.phoneNumbersCount << "\n";
        res << "CPU solving time:\t\t" << cpuSolveTimeMs << " ms\n";
        res << "GPU solving time:\t\t" << gpuSolveTimeMs << " ms\n";
        res.precision(2);
        res << ocl::utils::filler(fillerSize, '-');
        if ((gpuSolveTimeMs >= cpuSolveTimeMs) && gpuSolveTimeMs != 0.0f)
            res << OCL_MAKE_CYAN("CPU") << " overwhelming percentage:\t" << 100*(gpuSolveTimeMs/cpuSolveTimeMs - 1) << "%\n";
        else if ((gpuSolveTimeMs < cpuSolveTimeMs) && cpuSolveTimeMs != 0.0f)
            res << OCL_MAKE_GREEN("GPU") << " overwhelming percentage:\t" << 100*(cpuSolveTimeMs/gpuSolveTimeMs - 1) << "%\n";
        res << ocl::utils::filler(fillerSize, '=');
        res << "\n";

        std::cout << res.str();
    }

private:
    void bufferToResults() {
        gpuResults.reserve(validationSet.size());
        for (size_t i = 0; i < _problem.phoneNumbersCount; i++) {
            if (numbersBuffer[i*12] == '#')
                continue;
            else if (numbersBuffer[i*12] == '+') {
                std::string result = "+";
                for (size_t j = i*12 + 1; j < 12*(i + 1); j++) {
                    result += numbersBuffer[j];
                }
                gpuResults.push_back(result);
            }
        }
    }

private:
    Lab2(const Lab2& other) {}
    Lab2& operator=(const Lab2& other) {}
    Lab2(Lab2&& other) noexcept {}
    Lab2& operator=(const Lab2&& other) noexcept {}
};
