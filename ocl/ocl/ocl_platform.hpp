#pragma once

#include "ocl_utility.hpp"

namespace ocl {

    namespace platform {

        static cl_uint num;
        static cl_platform_id* platforms;

        bool init() {
            cl_int err = clGetPlatformIDs(0, 0, &num);
            if (err != CL_SUCCESS || num == 0) {
                OCL_LOG_ERROR << "OpenCL platforms are not detected (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Detected " << num << " OpenCL platforms\n";

            if (platforms != nullptr) {
                delete[] platforms;
                platforms = nullptr;
            }

            platforms = new cl_platform_id[num];

            err = clGetPlatformIDs(num, platforms, 0);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "clGetPlatformIDs returned an error (err=" << err << ")\n";
                return false;
            }

            return true;
        }

        bool checkInit(const std::string& callerInfo = "Platform") {
            if (platforms == nullptr) {
                log::stream() << OCL_MAKE_YELLOW(callerInfo) << ": Platform should be initialized first\n";
                return false;
            }

            return true;
        }

        void printInfo() {
            if (!checkInit(__FUNCTION__))
                return;

            if (num == 0) {
                OCL_LOG_ERROR << "Print failed because of 0 platforms\n";
                return;
            }
            OCL_LOG_DEFAULT << "Printing information about " << num << " platforms:\n";
            const int strbufSize = 250;
            char strbuf[strbufSize];

            cl_platform_id* pPtr = platforms;
            if (pPtr == nullptr) {
                OCL_LOG_ERROR << "Print failed becuase platforms pointer was nullptr\n";
                return;
            }

            auto printLambda = [&pPtr, &strbuf, strbufSize](int pId, cl_uint cl_parameter, const std::string& name) {
                clGetPlatformInfo(platforms[pId], cl_parameter, strbufSize, (void*)strbuf, 0);
                ocl::log::stream() << "\t" << OCL_MAKE_YELLOW(name) << ":" << utils::getTab(name.size()) << strbuf << "\n";
            };

            for (unsigned int i = 0; i < num; i++) {
                ocl::log::stream() << "P#" << i + 1 << " ";
                printLambda(i, CL_PLATFORM_PROFILE,     "PROFILE");
                printLambda(i, CL_PLATFORM_VERSION,     "VERSION");
                printLambda(i, CL_PLATFORM_NAME,        "NAME");
                printLambda(i, CL_PLATFORM_VENDOR,      "VENDOR");
                printLambda(i, CL_PLATFORM_EXTENSIONS,  "EXTENSIONS");
                if (i != num - 1)
                    ocl::log::stream() << "\n";
            }
        }

        bool cleanup() {
            if (!checkInit())
                return true;

            if (platforms != nullptr) {
                try {
                    delete[] platforms;
                    OCL_LOG_POSITIVE << "Platforms released successfully\n";
                }
                catch (std::exception& e) {
                    OCL_LOG_ERROR << "Trouble releasing platforms (" << e.what() << ")\n";
                    return false;
                }
                platforms = nullptr;
            }
            else
                return true;

            return true;
        }

    } // namespace platform

} // namespace ocl
