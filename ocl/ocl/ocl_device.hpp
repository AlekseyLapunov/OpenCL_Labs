#pragma once

#include "ocl_utility.hpp"
#include "ocl_platform.hpp"

namespace ocl {

    namespace device {

        static cl_uint num;
        static cl_device_id* devices;

        bool checkInit(const std::string& callerInfo = "Device") {
            if (devices == nullptr) {
                ocl::log::stream() << OCL_MAKE_YELLOW(callerInfo) << ": Device should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup() {
            if (!checkInit())
                return true;

            if (devices != nullptr) {
                try {
                    delete[] devices;
                    OCL_LOG_POSITIVE << "Devices released successfully\n";
                }
                catch (std::exception& e) {
                    OCL_LOG_ERROR << "Trouble releasing devices (" << e.what() << ")\n";
                    return false;
                }
                devices = nullptr;
            }
            else
                return true;

            return true;
        }

        bool init() {
            if (!platform::checkInit(__FUNCTION__)) {
                return false;
            }

            cl_int err = clGetDeviceIDs(platform::platforms[0], CL_DEVICE_TYPE_ALL, 0, 0, &num);
            if (err != CL_SUCCESS || num == 0) {
                OCL_LOG_ERROR << "OpenCL devices are not detected (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Detected " << num << " OpenCL devices on the first platform\n";

            if (devices != nullptr)
                cleanup();

            devices = new cl_device_id[num];

            err = clGetDeviceIDs(platform::platforms[0], CL_DEVICE_TYPE_ALL, num, devices, 0);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "clGetDeviceIDs returned an error (err=" << err << ")\n";
                return false;
            }

            return true;
        }

        void printInfo() {
            if (!platform::checkInit(__FUNCTION__))
                return;

            if (platform::num == 0) {
                OCL_LOG_ERROR << "Print failed because of 0 platforms.\n";
                return;
            }

            OCL_LOG_DEFAULT << "Printing information about " << num << " devices:\n";
            cl_uint arg;
            const int strbufSize = 250;
            char strbuf[strbufSize];

            cl_device_id* dPtr = devices;
            if (dPtr == nullptr) {
                OCL_LOG_ERROR << "Print failed because devices pointer was nullptr\n";
                return;
            }

            auto printStrLambda = [&dPtr, &strbuf, strbufSize](int dId, cl_uint cl_parameter, const std::string& name) {
                clGetDeviceInfo(devices[dId], cl_parameter, strbufSize, (void*)strbuf, 0);
                ocl::log::stream() << "\t" << OCL_MAKE_YELLOW(name) << ":" << utils::getTab(name.size()) << strbuf << "\n";
            };
            auto printArgLambda = [&dPtr, arg](int dId, cl_uint cl_parameter, const std::string& name) {
                clGetDeviceInfo(devices[dId], cl_parameter, sizeof(arg), (void*)&arg, 0);
                ocl::log::stream() << "\t" << OCL_MAKE_YELLOW(name) << ":" << utils::getTab(name.size()) << arg << "\n";
            };

            for (unsigned int i = 0; i < num; i++) {
                ocl::log::stream() << "D#" << i + 1 << " ";
                printStrLambda(i, CL_DEVICE_NAME,                   "NAME");
                printStrLambda(i, CL_DEVICE_OPENCL_C_VERSION,       "OPENCL_VERSION");
                printArgLambda(i, CL_DEVICE_MAX_CLOCK_FREQUENCY,    "MAX_CLOCK_FREQUENCY");
                printArgLambda(i, CL_DEVICE_MAX_COMPUTE_UNITS,      "MAX_COMPUTE_UNITS");
                if (i != num - 1)
                    ocl::log::stream() << "\n";
            }
        }

    } // namespace device

} // namespace ocl
