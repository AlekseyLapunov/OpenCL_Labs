#pragma once

#include "ocl_device.hpp"
#include "ocl_program.hpp"
#include "ocl_memory.hpp"
#include "ocl_utility.hpp"

namespace ocl {

    namespace kernel {

        static bool isInit = false;
        static cl_kernel kernel;
        static cl_uint argCount = 0;

        bool init(const std::string& kernelFunctionName, std::ostream& log) {
            isInit = false;

            if (!program::checkInit(log, __FUNCTION__))
                return false;

            cl_int err;
            const char* functionNameArg = kernelFunctionName.c_str();
            kernel = clCreateKernel(program::program, functionNameArg, &err);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Creating of kernel has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Kernel created successfully\n";

            argCount = 0;
            isInit = true;
            return true;
        }

        bool checkInit(std::ostream& log, const std::string& callerInfo = "Kernel") {
            if (!isInit) {
                log << MAKE_YELLOW(callerInfo) << ": Kernel should be initialized first\n";
                return false;
            }

            return true;
        }
        
        void emplaceArgument(void* arg, size_t argTypeSize, std::ostream& log = std::cout) {
            if (!program::checkInit(log, __FUNCTION__) || !checkInit(log, __FUNCTION__))
                return;

            if (!memory::writeArgument(arg, argTypeSize, log)) {
                log << MAKE_RED(__FUNCTION__) << ": Writing the argument into device memory has failed\t" << "[pos " << argCount << "]\n";
                return;
            }

            cl_int err = clSetKernelArg(ocl::kernel::kernel, argCount, sizeof(cl_mem),
                (void*)&ocl::memory::getBufferByArgPos(argCount).first);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": clSetKernelArg() returned error (err=" << err << ")\t" << "[pos " << argCount << "]\n";
                return;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Successfully set argument\t\t\t" << "[pos " << argCount << "]\n";

            argCount++;
        }

        void emplaceArgumentArray(void* argArr, size_t argTypeSize, size_t argArrSize, std::ostream& log = std::cout) {
            if (!program::checkInit(log, __FUNCTION__) || !checkInit(log, __FUNCTION__))
                return;

            emplaceArgument(argArr, argTypeSize*argArrSize, log);
        }

        void printInfo(std::ostream& log) {
            if (!device::checkInit(log, __FUNCTION__) || !checkInit(log, __FUNCTION__))
                return;

            size_t arg;
            cl_int err = clGetKernelWorkGroupInfo(kernel::kernel, device::devices[0],
                CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void*)&arg, NULL);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Call to clGetKernelWorkGroupInfo() resulted in error (fetching CL_KERNEL_WORK_GROUP_SIZE) (err=" << err << ")\n";
                return;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Maximum count of working elements inside one work group : " << arg << "\n";

            err = clGetKernelWorkGroupInfo(kernel::kernel, device::devices[0],
                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                sizeof(size_t), (void*)&arg, NULL);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Call to clGetKernelWorkGroupInfo() resulted in error (fetching CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE) (err=" << err << ")\n";
                return;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Recommended common multiple of working elements inside one work group: " << arg << "\n";
        }

        bool cleanup(std::ostream& log) {
            if (!isInit)
                return true;

            isInit = false;
            cl_int err = clReleaseKernel(kernel);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Call to clReleaseKernel() has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Kernel released successfully\n";

            return true;
        }

    } // namespace kernel

} // namespace ocl
