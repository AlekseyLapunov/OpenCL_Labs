#pragma once

#include "ocl_utility.hpp"
#include "ocl_context.hpp"

namespace ocl {

    namespace command_queue {

        bool isInit = false;
        static cl_command_queue queue;

        bool init(std::ostream& log) {
            isInit = false;

            if (!context::checkInit(log, __FUNCTION__))
                return false;

            cl_int err;
            queue = clCreateCommandQueue(context::context, device::devices[0], 0, &err);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Creating of queue has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Queue created successfully on the first device\n";

            isInit = true;
            return true;
        }

        bool checkInit(std::ostream& log, const std::string& callerInfo = "Command Queue") {
            if (!isInit) {
                log << MAKE_YELLOW(callerInfo) << ": Command Queue should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup(std::ostream& log) {
            if (!isInit)
                return true;

            isInit = false;
            cl_int err = clReleaseCommandQueue(queue);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Call to clReleaseCommandQueue() has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Command Queue released successfully\n";

            return true;
        }

    } // namespace command_queue

} // namespace ocl
