#pragma once

#include "ocl_utility.hpp"

namespace ocl {

    namespace platform {

        static bool isInit = false;
        static cl_uint num;
        static cl_platform_id* platforms;

        bool init(std::ostream& log) {
            isInit = false;

            cl_int err = clGetPlatformIDs(0, 0, &num);
            if (err == CL_SUCCESS && num != 0) {
                log << MAKE_GREEN(__FUNCTION__) << ": Detected " << num << " OpenCL platforms\n";
            }
            else {
                log << MAKE_RED(__FUNCTION__) << ": OpenCL platforms are not detected (err=" << err << ")\n";
                return false;
            }

            if (platforms != nullptr) {
                delete[] platforms;
                platforms = nullptr;
            }

            platforms = new cl_platform_id[num];

            err = clGetPlatformIDs(num, platforms, 0);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": clGetPlatformIDs returned an error (err=" << err << ")\n";
                return false;
            }

            isInit = true;
            return true;
        }

        bool checkInit(std::ostream& log, const std::string& callerInfo = "Platform") {
            if (!isInit || platforms == nullptr) {
                log << MAKE_YELLOW(callerInfo) << ": Platform should be initialized first\n";
                return false;
            }

            return true;
        }

        void printInfo(std::ostream& log) {
            if (!checkInit(log, __FUNCTION__))
                return;

            if (num == 0) {
                log << MAKE_RED(__FUNCTION__) << ": Print failed because of 0 platforms\n";
                return;
            }

            log << MAKE_GREEN(__FUNCTION__) << ": Printing information about " << num << " platforms:\n";
            const int strbufSize = 250;
            char strbuf[strbufSize];

            cl_platform_id* pPtr = platforms;
            if (pPtr == nullptr) {
                log << MAKE_RED(__FUNCTION__) << ": Print failed becuase platforms pointer was nullptr\n";
                return;
            }

            auto printLambda = [&pPtr, &strbuf, strbufSize, &log](int pId, cl_uint cl_parameter, const std::string& name) {
                clGetPlatformInfo(platforms[pId], cl_parameter, strbufSize, (void*)strbuf, 0);
                log << "\t" << MAKE_YELLOW(name) << ":" << utils::getTab(name.size()) << strbuf << "\n";
            };

            for (unsigned int i = 0; i < num; i++) {
                log << "P#" << i + 1 << " ";
                printLambda(i, CL_PLATFORM_PROFILE,     "PROFILE");
                printLambda(i, CL_PLATFORM_VERSION,     "VERSION");
                printLambda(i, CL_PLATFORM_NAME,        "NAME");
                printLambda(i, CL_PLATFORM_VENDOR,      "VENDOR");
                printLambda(i, CL_PLATFORM_EXTENSIONS,  "EXTENSIONS");
                if (i != num - 1)
                    log << "\n";
            }
        }

        bool cleanup(std::ostream& log) {
            if (!isInit)
                return true;

            if (platforms != nullptr) {
                try {
                    delete[] platforms;
                    log << MAKE_GREEN(__FUNCTION__) << ": platforms released successfully\n";
                }
                catch (std::exception& e) {
                    log << MAKE_RED(__FUNCTION__) << ": trouble releasing platforms (" << e.what() << ")\n";
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
