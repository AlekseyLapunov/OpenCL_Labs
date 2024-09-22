#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <CL/cl.h>

#define OCL_MAKE_RED(input)      ("\033[91m" + std::string(input) + "\033[0m")
#define OCL_MAKE_YELLOW(input)   ("\033[93m" + std::string(input) + "\033[0m")
#define OCL_MAKE_GREEN(input)    ("\033[92m" + std::string(input) + "\033[0m")
#define OCL_MAKE_CYAN(input)     ("\033[96m" + std::string(input) + "\033[0m")

#define OCL_LOG_ERROR            ocl::log::stream() << OCL_MAKE_RED(__FUNCTION__)    << ":\t"
#define OCL_LOG_WARNING          ocl::log::stream() << OCL_MAKE_YELLOW(__FUNCTION__) << ":\t"
#define OCL_LOG_POSITIVE         ocl::log::stream() << OCL_MAKE_GREEN(__FUNCTION__)  << ":\t"
#define OCL_LOG_DEFAULT          ocl::log::stream() << OCL_MAKE_CYAN(__FUNCTION__)   << ":\t"

#define OCL_INIT_WARNING(caller) ocl::log::stream() << OCL_MAKE_YELLOW(caller) << ":\t"

namespace ocl {

    namespace log {

        static std::ostream* _stream = &std::cout;

        inline std::ostream& stream() {
            return *_stream;
        }

        void setLogStream(std::ostream& log) {
            _stream = &log;
        }

    } // namespace log

    namespace utils {

        static size_t fillerLength = 80;
        static char   fillerSign   = '-';

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
            indent.resize(size);
            for (int i = 0; i < size; i++) {
                indent += '\n';
            }

            return indent;
        }

        std::string readFileIntoString(const std::string& fileName) {
            std::ifstream fileStream{fileName};

            if (!fileStream.is_open()) {
                OCL_LOG_ERROR << "Can not open file \"" << fileName << "\"\n";
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

        std::string filler(size_t length = fillerLength, char sign = fillerSign) {
            std::string output;
            output.resize(length + 1);
            for (size_t i = 0; i < length; i++)
                output += sign;
            output += '\n';
            return output;
        }

        std::string fillerWithFileName(const std::string& fileName = "file", size_t length = fillerLength, char sign = fillerSign) {
            if (fileName.size() > length)
                return fileName;

            if (fileName.empty())
                return filler(length, sign);

            std::string output;
            output.resize(length + 1);
            size_t sideLen = (length - fileName.size()) / 2;
            for (size_t left = 0; left < sideLen - 2; left++)
                output += sign;
            output += "[ ";
            output += fileName;
            output += " ]";
            for (size_t right = 0; right < sideLen - 2; right++)
                output += sign;
            output += '\n';
            return output;
        }

    } // namespace utils

    namespace callback {
        static std::string sourceFileNameMemo;

        void __stdcall builder(cl_program program, void* userData) {
            cl_device_id device = *(cl_device_id*)userData;
            const int strSize = 2000;
            char bufStr[strSize];
            cl_int err;
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, strSize, (void*)bufStr, NULL);

            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "\tCalling to clGetProgramBuildInfo() resulted in error (err=" << err << ")\n";
                return;
            }

            if (bufStr[0] == '\0' || bufStr[0] == '\n') {
                return;
            }

            OCL_LOG_WARNING << "\tEvent \"clBuildProgram\"\n";
            OCL_LOG_DEFAULT << "\tOpenCL Builder log next lines:\n";
            ocl::log::stream() << OCL_MAKE_YELLOW(ocl::utils::fillerWithFileName(sourceFileNameMemo));
            ocl::log::stream() << OCL_MAKE_CYAN(bufStr);
            ocl::log::stream() << OCL_MAKE_YELLOW(ocl::utils::filler());
        }

        void __stdcall contextNotify(const char* errinfo, const void* private_info, size_t cv, void* user_data) {
            if (errinfo == nullptr)
                return;

            if (errinfo[0] == '\0' || errinfo[0] == '\n') {
                return;
            }

            OCL_LOG_WARNING << "\tEvent \"clCreateContext\"\n";
            OCL_LOG_DEFAULT << "\tOpenCL context pfn_notify errinfo next lines:\n\t" << errinfo;
        }

    } // namespace callbacks

} // namespace ocl