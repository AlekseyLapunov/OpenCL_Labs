#pragma once

#include "ocl_device.hpp"
#include "ocl_program.hpp"
#include "ocl_memory.hpp"
#include "ocl_utility.hpp"

namespace ocl {

    namespace kernel {

        static cl_kernel kernel;
        static cl_uint argCount = 0;

        bool init(const std::string& kernelFunctionName) {
            if (!program::checkInit(__FUNCTION__))
                return false;

            cl_int err;
            const char* functionNameArg = kernelFunctionName.c_str();
            kernel = clCreateKernel(program::program, functionNameArg, &err);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Creating of kernel has failed(err = " << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Kernel created successfully\n";

            argCount = 0;
            return true;
        }

        bool checkInit(const std::string& callerInfo = "Kernel") {
            if (kernel == nullptr) {
                ocl::log::stream() << OCL_MAKE_YELLOW(callerInfo) << ": Kernel should be initialized first\n";
                return false;
            }

            return true;
        }
        
        void emplaceArgument(void* arg, size_t argTypeSize) {
            if (!program::checkInit(__FUNCTION__) || !checkInit(__FUNCTION__))
                return;

            if (!memory::writeArgument(arg, argTypeSize)) {
                OCL_LOG_ERROR << "Writing the argument into device memory has failed\t" << "[pos " << argCount << "]\n";
                return;
            }

            cl_int err = clSetKernelArg(ocl::kernel::kernel, argCount, sizeof(cl_mem),
                (void*)&ocl::memory::getBufferByArgPos(argCount).first);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "clSetKernelArg() returned error (err=" << err << ")\t" << "[pos " << argCount << "]\n";
                return;
            }
            OCL_LOG_POSITIVE << "Successfully set argument\t\t" << "[pos " << argCount << "]\n";

            argCount++;
        }

        void emplaceArgumentArray(void* argArr, size_t argTypeSize, size_t argArrSize) {
            if (!program::checkInit(__FUNCTION__) || !checkInit(__FUNCTION__))
                return;

            emplaceArgument(argArr, argTypeSize*argArrSize);
        }

        void printInfo() {
            if (!device::checkInit(__FUNCTION__) || !checkInit(__FUNCTION__))
                return;

            OCL_LOG_DEFAULT << "Printing Kernel information:\n";

            size_t arg;
            cl_int err = clGetKernelWorkGroupInfo(kernel::kernel, device::devices[0],
                CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void*)&arg, NULL);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Call to clGetKernelWorkGroupInfo() resulted in error (fetching CL_KERNEL_WORK_GROUP_SIZE) (err=" << err << ")\n";
                return;
            }
            OCL_LOG_POSITIVE << "Maximum count of working elements inside one work group : " << arg << "\n";

            err = clGetKernelWorkGroupInfo(kernel::kernel, device::devices[0],
                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                sizeof(size_t), (void*)&arg, NULL);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Call to clGetKernelWorkGroupInfo() resulted in error (fetching CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE) (err=" << err << ")\n";
                return;
            }
            OCL_LOG_POSITIVE << "Recommended common multiple of working elements inside one work group: " << arg << "\n";
        }

        bool cleanup() {
            if (!checkInit())
                return true;

            cl_int err = clReleaseKernel(kernel);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Call to clReleaseKernel() has failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Kernel released successfully\n";

            return true;
        }

    } // namespace kernel

} // namespace ocl
