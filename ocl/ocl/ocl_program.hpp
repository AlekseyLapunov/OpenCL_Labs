#pragma once

#include "ocl_utility.hpp"
#include "ocl_context.hpp"

namespace ocl {

    namespace program {

        static bool         isInit = false;
        static cl_program   program;
        static std::string  source;

        void printSource(std::ostream& log) {
            if (source.empty())
                return;

            log << MAKE_GREEN(__FUNCTION__) << ": Printing OpenCL source code next lines:\n";
            log << "\t" << MAKE_CYAN(source);
        }

        bool init(const std::string& programSource, std::ostream& log) {
            isInit = false;

            if (!context::checkInit(log, __FUNCTION__))
                return false;

            cl_int err;
            source = programSource;
            const char* argSource = source.c_str();
            program = clCreateProgramWithSource(context::context, 1, &argSource, NULL, &err);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Creating of program has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Program created successfully. Compiling the source code next...\n";

            err = clBuildProgram(program, 1, device::devices, NULL, callback::builder, (void*)device::devices);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Source code compilation failed. Callback function should print builder logs (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Source code compiled successfully\n";

            isInit = true;
            return true;
        }

        bool checkInit(std::ostream& log, const std::string& callerInfo = "Program") {
            if (!isInit) {
                log << MAKE_YELLOW(callerInfo) << ": Program should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup(std::ostream& log) {
            if (!isInit)
                return true;

            isInit = false;
            cl_int err = clReleaseProgram(program);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Call to clReleaseProgram() has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Program released successfully\n";

            return true;
        }

    } // namespace program

} // namespace ocl
