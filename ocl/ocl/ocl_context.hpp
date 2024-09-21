#pragma once

#include "ocl_utility.hpp"
#include "ocl_platform.hpp"
#include "ocl_device.hpp"

namespace ocl {

    namespace context {

        static cl_context context;

        bool checkInit(const std::string& callerInfo = "Context") {
            if (context == nullptr) {
                ocl::log::stream() << OCL_MAKE_YELLOW(callerInfo) << ": Context should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup() {
            if (!checkInit())
                return true;

            cl_int err = clReleaseContext(context);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Call to clReleaseContext() has failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Context released successfully\n";

            return true;
        }

        bool init() {
            if (!platform::checkInit(__FUNCTION__) || !device::checkInit(__FUNCTION__))
                return false;

            if (context != nullptr)
                cleanup();

            cl_context_properties properties[3] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform::platforms[0], 0 };

            cl_int err;
            context = clCreateContext(properties, device::num, device::devices,
                callback::contextNotify, NULL, &err);
            if (err != CL_SUCCESS) {
                OCL_LOG_ERROR << "Creating failed (err=" << err << ")\n";
                return false;
            }
            OCL_LOG_POSITIVE << "Context created successfully\n";

            return true;
        }

    } // namespace context

} // namespace ocl
