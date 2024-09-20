#pragma once

#include "ocl_utility.hpp"
#include "ocl_platform.hpp"

namespace ocl {

    namespace device {

        static cl_uint num;
        static cl_device_id* devices;
        static bool isInit = false;

        bool init(std::ostream& log) {
            isInit = false;

            if (!platform::checkInit(log, __FUNCTION__)) {
                return false;
            }

            cl_int err = clGetDeviceIDs(platform::platforms[0], CL_DEVICE_TYPE_ALL, 0, 0, &num);
            if (err == CL_SUCCESS && num != 0) {
                log << MAKE_GREEN(__FUNCTION__) << ": Detected " << num << " OpenCL devices on the first platform\n";
            }
            else {
                log << MAKE_RED(__FUNCTION__) << ": OpenCL devices are not detected (err=" << err << ")\n";
                return false;
            }

            if (devices != nullptr) {
                delete[] devices;
                devices = nullptr;
            }

            devices = new cl_device_id[num];

            err = clGetDeviceIDs(platform::platforms[0], CL_DEVICE_TYPE_ALL, num, devices, 0);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": clGetDeviceIDs returned an error (err=" << err << ")\n";
                return false;
            }

            isInit = true;
            return true;
        }

        bool checkInit(std::ostream& log, const std::string& callerInfo = "Device") {
            if (!isInit || devices == nullptr) {
                log << MAKE_YELLOW(__FUNCTION__) << ": Device should be initialized first\n";
                return false;
            }

            return true;
        }

        void printInfo(std::ostream& log) {
            if (!platform::checkInit(log, __FUNCTION__))
                return;

            if (platform::num == 0) {
                log << MAKE_RED(__FUNCTION__) << ": Print failed because of 0 platforms.\n";
                return;
            }

            log << MAKE_GREEN(__FUNCTION__) << ": Printing information about " << num << " devices:\n";
            cl_uint arg;
            const int strbufSize = 250;
            char strbuf[strbufSize];

            cl_device_id* dPtr = devices;
            if (dPtr == nullptr) {
                log << MAKE_RED(__FUNCTION__) << ": Print failed because devices pointer was nullptr\n";
                return;
            }

            auto printStrLambda = [&dPtr, &strbuf, strbufSize, &log](int dId, cl_uint cl_parameter, const std::string& name) {
                clGetDeviceInfo(devices[dId], cl_parameter, strbufSize, (void*)strbuf, 0);
                log << "\t" << MAKE_YELLOW(name) << ":" << utils::getTab(name.size()) << strbuf << "\n";
            };
            auto printArgLambda = [&dPtr, arg, &log](int dId, cl_uint cl_parameter, const std::string& name) {
                clGetDeviceInfo(devices[dId], cl_parameter, sizeof(arg), (void*)&arg, 0);
                log << "\t" << MAKE_YELLOW(name) << ":" << utils::getTab(name.size()) << arg << "\n";
            };

            for (unsigned int i = 0; i < num; i++) {
                log << "D#" << i + 1 << " ";
                printStrLambda(i, CL_DEVICE_NAME,                   "NAME");
                printStrLambda(i, CL_DEVICE_OPENCL_C_VERSION,       "OPENCL_VERSION");
                printArgLambda(i, CL_DEVICE_MAX_CLOCK_FREQUENCY,    "MAX_CLOCK_FREQUENCY");
                printArgLambda(i, CL_DEVICE_MAX_COMPUTE_UNITS,      "MAX_COMPUTE_UNITS");
                if (i != num - 1)
                    log << "\n";
            }
        }

        bool cleanup(std::ostream& log) {
            if (!isInit)
                return true;

            if (devices != nullptr) {
                try {
                    delete[] devices;
                    log << MAKE_GREEN(__FUNCTION__) << ": devices released successfully\n";
                }
                catch (std::exception& e) {
                    log << MAKE_RED(__FUNCTION__) << ": trouble releasing devices (" << e.what() << ")\n";
                    return false;
                }
                devices = nullptr;
            }
            else
                return true;

            return true;
        }

    } // namespace device

} // namespace ocl
