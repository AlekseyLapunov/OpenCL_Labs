#pragma once

#include "ocl_utility.hpp"
#include "ocl_platform.hpp"
#include "ocl_device.hpp"
#include "ocl_context.hpp"
#include "ocl_command_queue.hpp"
#include "ocl_program.hpp"
#include "ocl_kernel.hpp"
#include "ocl_memory.hpp"

namespace ocl {

    bool init(const std::string& sourceFileName, const std::string& kernelFunctionName) {
        if (!platform::init()) return false;
        if (!device::init()) return false;
        if (!context::init()) return false;
        if (!command_queue::init()) return false;
        program::sourceFileName = sourceFileName;
        callback::sourceFileNameMemo = sourceFileName;
        if (!program::init(utils::readFileIntoString(program::sourceFileName))) return false;
        if (!kernel::init(kernelFunctionName)) return false;
        return true;
    }

    template <typename T>
    void addArgument(T& arg) {
        ocl::kernel::emplaceArgument((void*)&arg, sizeof(T));
    }

    template <typename T>
    void addArgumentArray(T& arr, size_t arrSize) {
        ocl::kernel::emplaceArgumentArray((void*)&arr, sizeof(T), arrSize);
    }

    KernelParameters& kernelPrompt(std::istream& ist = std::cin) {
        ocl::log::stream() << "Enter working dimension >> ";
        ist >> ocl::_kernelParameters.dimension;
        ist.clear();

        if (_kernelParameters.dimension == 0 || _kernelParameters.dimension > 3) {
            ocl::log::stream() << "Wrong dimension value (1 <= dim <= 3), using dim=1.\n";
            _kernelParameters.dimension = 1;
        }

        for (size_t i = 1; i <= _kernelParameters.dimension; i++) {
            ocl::log::stream() << "Enter global work size (" << i << "/" << _kernelParameters.dimension << ") >> ";
            ist >> _kernelParameters.gWorkSize[i - 1];
            ist.clear();
        }

        char autoSplitChar = ' ';
        ocl::log::stream() << "Use auto split? (y/n) >> ";
        ist >> autoSplitChar;
        if (autoSplitChar == 'y')
            ocl::_kernelParameters.autoSplit = true;
        else if (autoSplitChar == 'n')
            ocl::_kernelParameters.autoSplit = false;
        else {
            ocl::log::stream() << "Unrecognized symbol, using auto split.\n";
            ocl::_kernelParameters.autoSplit = true;
        }

        if (_kernelParameters.autoSplit)
            return _kernelParameters;

        for (size_t i = 1; i <= _kernelParameters.dimension; i++) {
            ocl::log::stream() << "Enter local work size (" << i << "/" << _kernelParameters.dimension << ") >> ";
            ist >> _kernelParameters.lWorkSize[i - 1];
            ist.clear();
        }

        return _kernelParameters;
    }

    template <typename T>
    bool executeKernel(cl_uint dim, const size_t globalWorkSize[], bool autoSplit, const size_t localWorkSize[],
                       size_t resArgPos, T& resBuf) {
        if (!kernel::checkInit(__FUNCTION__))
            return false;

        if (dim == 0 || dim > 3) {
            OCL_LOG_ERROR << "Wrong dimension value (1 <= dim <= 3)\n";
            return false;
        }

        OCL_LOG_POSITIVE << "Starting kernel for execution\n";
        cl_int err = clEnqueueNDRangeKernel(command_queue::queue, kernel::kernel,
            dim, NULL, globalWorkSize, (autoSplit ? NULL : localWorkSize), 0, 0, 0);
        if (err != CL_SUCCESS) {
            OCL_LOG_ERROR << "Error starting the kernel (err=" << err << ")\n";
            return false;
        }

        err = clFinish(command_queue::queue);
        if (err != CL_SUCCESS) {
            OCL_LOG_ERROR << "Kernel finish has failed (err=" << err << ")\n";
            return false;
        }
        OCL_LOG_POSITIVE << "Kernel finished the calculations successfuly\n";

        if (!memory::readByArgPos(resArgPos, (void*)&resBuf)) {
            OCL_LOG_ERROR << "Failed to fetch result of kernel execution\n";
            return false;
        }
        else {
            OCL_LOG_POSITIVE << "Result of kernel execution successfully fetched\n";
            return true;
        }
    }

    template <typename T>
    bool promptExecuteKernel(size_t resArgPos, T& resBuf, std::istream& promptStream = std::cin) {
        KernelParameters& kp = kernelPrompt(promptStream);
        return executeKernel(kp.dimension, kp.gWorkSize, kp.autoSplit, kp.lWorkSize, resArgPos, resBuf);
    }

    void printVerboseInfo(uint8_t indent = 0) {
        ocl::log::stream() << utils::getIndent(indent);
        platform::printInfo();
        device::printInfo();
        program::printSource();
        kernel::printInfo();
        ocl::log::stream() << utils::getIndent(indent);
    }

    void cleanup() {
        size_t troubleCounter = 0;
        const size_t components = 7;

        OCL_LOG_DEFAULT << "Starting clean-up...\n";
        if (!platform::cleanup())      ++troubleCounter;
        if (!device::cleanup())        ++troubleCounter;
        if (!context::cleanup())       ++troubleCounter;
        if (!command_queue::cleanup()) ++troubleCounter;
        if (!program::cleanup())       ++troubleCounter;
        if (!kernel::cleanup())        ++troubleCounter;
        if (!memory::cleanup())        ++troubleCounter;

        if (troubleCounter == 0)
            OCL_LOG_POSITIVE << "Clean-up finished successfully: released created components\n";
        else
            OCL_LOG_WARNING << troubleCounter << " of " << components << " components were not cleaned up properly.\n";
    }

} // namespace ocl
