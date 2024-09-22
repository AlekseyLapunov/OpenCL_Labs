#pragma once

#include "ocl_utility.hpp"
#include "ocl_context.hpp"

namespace ocl {

    namespace command_queue {

        static cl_command_queue queue;

        bool checkInit(const std::string& callerInfo = "Command Queue") {
            if (queue == nullptr) {
                OCL_INIT_WARNING(callerInfo) << "Command Queue should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup() {
            if (!checkInit())
                return true;

            cl_int err = clReleaseCommandQueue(queue);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Call to clReleaseCommandQueue() has failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Command Queue released successfully\n";

            return true;
        }

        bool init() {
            if (!context::checkInit(__FUNCTION__))
                return false;

            if (queue != nullptr)
                cleanup();

            cl_int err;
            queue = clCreateCommandQueue(context::context, device::devices[0], 0, &err);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Creating of queue has failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Queue created successfully on the first device\n";

            return true;
        }

    } // namespace command_queue

} // namespace ocl
