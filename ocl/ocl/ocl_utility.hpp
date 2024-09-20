#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <CL/cl.h>

#define MAKE_RED(input)     "\033[91m" << input << "\033[0m"
#define MAKE_YELLOW(input)  "\033[93m" << input << "\033[0m"
#define MAKE_GREEN(input)   "\033[92m" << input << "\033[0m"
#define MAKE_CYAN(input)    "\033[96m" << input << "\033[0m"

namespace ocl {

    namespace utils {

        std::string getTab(size_t size) {
            if (size <= 6)
                return "\t\t\t";
            if (size <= 16)
                return "\t\t";
            return "\t";
        }

        inline std::string getIndent(int size) {
            if (size <= 0)
                return "";

            std::string indent;
            indent.reserve(size);

            for (int i = 0; i < size; i++) {
                indent.push_back('\n');
            }

            return indent;
        }

        std::string readFileIntoString(const std::string& fileName, std::ostream& log) {
            std::ifstream fileStream{fileName};

            if (!fileStream.is_open()) {
                log << MAKE_RED(__FUNCTION__) << ": can't open file \"" << fileName << "\"\n";
            }

            std::string output;
            std::string line;
            fileStream.clear();
            fileStream.seekg(0, std::ios::beg);

            while (std::getline(fileStream, line)) {
                output += line;
                output.push_back('\n');
            }

            fileStream.close();

            return output;
        }

    } // namespace utils

    namespace callback {

        void __stdcall builder(cl_program program, void* userData) {
            std::cerr << "\tEvent \"clBuildProgram\"\n";

            cl_device_id device = *(cl_device_id*)userData;
            const int strSize = 2000;
            char bufStr[strSize];
            cl_int err;
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, strSize, (void*)bufStr, NULL);

            if (err != CL_SUCCESS) {
                std::cerr << "\tCalling to clGetProgramBuildInfo() resulted in error (err=" << err << ")\n";
                return;
            }

            if (bufStr[0] == '\0' || bufStr[0] == '\n') {
                std::cerr << "\tBuilder log is empty\n";
                return;
            }

            std::cerr << "\tOpenCL Builder log next lines:\n\t";
            std::cerr << bufStr << "\n";
        }

        void __stdcall contextNotify(const char* errinfo, const void* private_info, size_t cv, void* user_data) {
            std::cerr << "\tEvent \"clCreateContext\"\n";

            if (errinfo == nullptr)
                return;

            if (errinfo[0] == '\0' || errinfo[0] == '\n') {
                std::cerr << "\tOpenCL context pfn_notify errinfo is empty\n";
                return;
            }

            std::cerr << "\tOpenCL context pfn_notify errinfo next lines:\n\t" << errinfo;
        }

    } // namespace callbacks

} // namespace ocl