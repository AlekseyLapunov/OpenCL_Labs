#pragma once

#include "ocl_utility.hpp"
#include "ocl_platform.hpp"
#include "ocl_device.hpp"

namespace ocl {

    namespace context {

        static bool isInit = false;
        static cl_context context;

        bool init(std::ostream& log) {
            isInit = false;
            if (!platform::checkInit(log, __FUNCTION__) || !device::checkInit(log, __FUNCTION__))
                return false;

            cl_context_properties properties[3] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform::platforms[0], 0 };

            cl_int err;
            context = clCreateContext(properties, device::num, device::devices,
                callback::contextNotify, NULL, &err);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Creating failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Context created successfully\n";

            isInit = true;
            return true;
        }

        bool checkInit(std::ostream& log, const std::string& callerInfo = "Context") {
            if (!isInit) {
                log << MAKE_YELLOW(callerInfo) << ": Context should be initialized first\n";
                return false;
            }

            return true;
        }

        bool cleanup(std::ostream& log) {
            if (!isInit)
                return true;

            isInit = false;
            cl_int err = clReleaseContext(context);
            if (err != CL_SUCCESS) {
                log << MAKE_RED(__FUNCTION__) << ": Call to clReleaseContext() has failed (err=" << err << ")\n";
                return false;
            }
            log << MAKE_GREEN(__FUNCTION__) << ": Context released successfully\n";

            return true;
        }

    } // namespace context

} // namespace ocl
