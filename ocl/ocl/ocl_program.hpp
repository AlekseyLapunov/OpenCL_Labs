#pragma once

#include "ocl_utility.hpp"
#include "ocl_context.hpp"

namespace ocl {

    namespace program {

        static cl_program  program;
        static std::string source;
        static std::string sourceFileName;

        bool checkInit(const std::string& callerInfo = "Program") {
            if (program == nullptr) {
                OCL_INIT_WARNING(callerInfo) << "Program should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup() {
            if (!checkInit())
                return true;

            cl_int err = clReleaseProgram(program);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Call to clReleaseProgram() has failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Program released successfully\n";

            return true;
        }

        bool init(const std::string& programSource) {
            if (!context::checkInit(__FUNCTION__))
                return false;

            if (program != nullptr)
                cleanup();

            cl_int err;
            source = programSource;
            const char* argSource = source.c_str();
            program = clCreateProgramWithSource(context::context, 1, &argSource, NULL, &err);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Creating of program has failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Program created successfully. Compiling the source code next...\n";

            err = clBuildProgram(program, 1, device::devices, NULL, callback::builder, (void*)device::devices);
            if (err != CL_SUCCESS || source.empty()) {
                OCL_LOG_ERROR << "Source code compilation failed (err=" << err << ")"
                    << (err == 0 ? ". Source code was empty" : ". Builder callback function should have printed compilation logs") << "\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Source code compiled successfully\n";

            return true;
        }

        void printSource() {
            if (source.empty())
                return;

            OCL_LOG_DEFAULT << "Printing OpenCL source code next lines:\n";
            ocl::log::stream() << OCL_MAKE_YELLOW(ocl::utils::fillerWithFileName(sourceFileName));
            ocl::log::stream() << OCL_MAKE_CYAN(source);
            ocl::log::stream() << OCL_MAKE_YELLOW(ocl::utils::filler());
        }

    } // namespace program

} // namespace ocl
