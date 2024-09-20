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

    bool init(const std::string& sourceFileName, const std::string& kernelFunctionName, std::ostream& log = std::cout) {
        if (!platform::init(log)) return false;
        if (!device::init(log)) return false;
        if (!context::init(log)) return false;
        if (!command_queue::init(log)) return false;
        if (!program::init(utils::readFileIntoString(sourceFileName, log), log)) return false;
        if (!kernel::init(kernelFunctionName, log)) return false;
        return true;
    }

    bool executeKernel(cl_uint dim, size_t globalWorkSize[], bool autoSplit, size_t localWorkSize[],
                       size_t resArgPos, void* resBuf,
                       std::ostream& log = std::cout) {
        if (!kernel::checkInit(log, __FUNCTION__))
            return false;

        if (dim == 0 || dim > 3) {
            log << MAKE_RED(__FUNCTION__) << ": Wrong dimension value (1 <= dim <= 3)\n";
            return false;
        }

        cl_int err = clEnqueueNDRangeKernel(command_queue::queue, kernel::kernel,
            dim, NULL, globalWorkSize, (autoSplit ? NULL : localWorkSize), 0, 0, 0);
        if (err != CL_SUCCESS) {
            log << MAKE_RED(__FUNCTION__) << ": Error starting the kernel (err=" << err << ")\n";
            return false;
        }
        log << MAKE_GREEN(__FUNCTION__) << ": Kernel started for execution\n";

        err = clFinish(command_queue::queue);
        if (err != CL_SUCCESS) {
            log << MAKE_RED(__FUNCTION__) << ": Kernel finish has failed (err=" << err << ")\n";
            return false;
        }
        log << MAKE_GREEN(__FUNCTION__) << ": Kernel finished the calculations successfuly\n";

        if (!memory::readByArgPos(resArgPos, resBuf, std::cout)) {
            log << MAKE_RED(__FUNCTION__) << ": Failed to fetch result of kernel execution\n";
            return false;
        }
        else {
            log << MAKE_GREEN(__FUNCTION__) << ": Result of kernel execution successfully fetched\n";
            return true;
        }
    }

    void printVerboseInfo(uint8_t indent = 0, std::ostream& log = std::cout) {
        log << utils::getIndent(indent);
        platform::printInfo(log);
        device::printInfo(log);
        program::printSource(log);
        kernel::printInfo(log);
        log << utils::getIndent(indent);
    }

    void cleanup(std::ostream& log = std::cout) {
        size_t troubleCounter = 0;
        const size_t components = 7;

        log << MAKE_GREEN(__FUNCTION__) << ": Starting clean-up...\n";
        if (!platform::cleanup(log))      ++troubleCounter;
        if (!device::cleanup(log))        ++troubleCounter;
        if (!context::cleanup(log))       ++troubleCounter;
        if (!command_queue::cleanup(log)) ++troubleCounter;
        if (!program::cleanup(log))       ++troubleCounter;
        if (!kernel::cleanup(log))        ++troubleCounter;
        if (!memory::cleanup(log))        ++troubleCounter;

        if (troubleCounter == 0)
            log << MAKE_GREEN(__FUNCTION__) << ": Clean-up finished successfully: released created components\n";
        else
            log << MAKE_YELLOW(__FUNCTION__) << ": " << troubleCounter << " of " << components << " components were not cleaned up properly.\n";
    }

} // namespace ocl
