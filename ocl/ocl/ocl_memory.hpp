#pragma once

#include "ocl_command_queue.hpp"
#include "ocl_context.hpp"
#include "ocl_utility.hpp"

namespace ocl {

    // Singleton
    class memory {
    private:
        static memory m_instance;

        std::ostream* m_log;
        memory() : m_log(&std::cout) {}

    public:
        using buffer_t  = std::pair<cl_mem, size_t>;
        using buffers_t = std::vector<buffer_t>;

        static bool writeArgument(void* arg, size_t argTypeSize, std::ostream& log = std::cout) {
            std::ostream& logRef = i(log).log();

            if (!context::checkInit(logRef, __FUNCTION__) || !command_queue::checkInit(logRef, __FUNCTION__))
                return false;
            
            buffers_t& memGpuRef = i(log).memGPU();
            size_t resultSize = argTypeSize;
            cl_int err;
            cl_mem argument = clCreateBuffer(context::context, CL_MEM_READ_WRITE, resultSize, NULL, &err);
            if (err != CL_SUCCESS) {
                logRef << MAKE_RED(__FUNCTION__) << ": Creating of memory object has failed (err=" << err << ")\n";
                return false;
            }
            memGpuRef.push_back({argument, argTypeSize});

            err = clEnqueueWriteBuffer(command_queue::queue, argument, CL_TRUE, 0, resultSize, arg, 0, 0, 0);
            if (err != CL_SUCCESS) {
                logRef << MAKE_RED(__FUNCTION__) << ": Calling to clEnqueueWriteBuffer() (arg -> memGPU) has failed (err=" << err << ")\n";
                return false;
            }
            logRef << MAKE_GREEN(__FUNCTION__) << ": Argument enqueued on write successfully\t" << "[pos " << (memGpuRef.size() - 1) << "]\n";

            return true;
        }

        static bool writeArgumentArray(void* argArr, size_t argTypeSize, size_t argArrSize, std::ostream& log = std::cout) {
            std::ostream& logRef = i(log).log();

            if (!writeArgument(argArr, argTypeSize * argArrSize, log))
                return false;

            return true;
        }

        static bool readByArgPos(size_t argPos, void* targetBuf, std::ostream& log = std::cout) {
            std::ostream& logRef = i(log).log();

            if (!context::checkInit(logRef, __FUNCTION__) || !command_queue::checkInit(logRef, __FUNCTION__))
                return false;
            
            buffers_t& memGpuRef = i(log).memGPU();
            if (argPos >= memGpuRef.size()) {
                logRef  << MAKE_RED(__FUNCTION__) << ": Wrong argument position\t" << "[pos " << argPos << "]\n";
                return false;
            }

            cl_int err = clEnqueueReadBuffer(ocl::command_queue::queue, memGpuRef.at(argPos).first, CL_TRUE,
                0, memGpuRef.at(argPos).second, (void*) targetBuf, 0, 0, 0);
            if (err != CL_SUCCESS) {
                logRef << "[pos " << argPos << "] " << MAKE_RED(__FUNCTION__) << ": Call to clEnqueueReadBuffer() resulted in error (err=" << err << ")\n";
                return false;
            }

            return true;
        }

        static void readInto(cl_mem mGPU, void* targetBuf, size_t bufSize, std::ostream& log = std::cout) {
            std::ostream& logRef = i(log).log();

            if (!context::checkInit(logRef, __FUNCTION__) || !command_queue::checkInit(logRef, __FUNCTION__))
                return;

            cl_int err = clEnqueueReadBuffer(ocl::command_queue::queue, mGPU, CL_TRUE,
                0, bufSize, (void*)targetBuf, 0, 0, 0);
            if (err != CL_SUCCESS) {
                logRef << MAKE_RED(__FUNCTION__) << ": Call to clEnqueueReadBuffer() resulted in error (err=" << err << ")\n";
                return;
            }
        }

        static buffer_t& getBufferByArgPos(size_t argPos, std::ostream& log = std::cout) {
            std::ostream& logRef = i(log).log();

            if (!context::checkInit(logRef, __FUNCTION__) || !command_queue::checkInit(logRef, __FUNCTION__))
                return i(log).nullBuffer();

            buffers_t& memGpuRef = i(log).memGPU();
            if (argPos >= memGpuRef.size()) {
                logRef << MAKE_RED(__FUNCTION__) << ": Wrong argument position\t" << "[pos " << argPos << "]\n";
                return i(log).nullBuffer();;
            }

            return memGpuRef.at(argPos);
        }

        static size_t writtenArgsCount() {
            return raw_i().memGPU().size();
        }

        static bool cleanup(std::ostream& log = std::cout) {
            std::ostream& logRef = i(log).log();
            buffers_t& memGpuRef = i(log).memGPU();

            if (memGpuRef.empty())
                return true;

            size_t pos = 0;
            size_t troubleCounter = 0;
            for (buffer_t& buf : memGpuRef) {
                cl_int err = clReleaseMemObject(buf.first);
                if (err != CL_SUCCESS) {
                    logRef << MAKE_RED(__FUNCTION__) << ": Trouble releasing memory object (err=" << err << ")\t" << "[pos " << pos << "]\n";
                    troubleCounter++;
                }

                pos++;
            }

            if (troubleCounter == 0)
                logRef << MAKE_GREEN(__FUNCTION__) << ": All " << memGpuRef.size() << " objects released successfuly.\n";
            else
                logRef << MAKE_YELLOW(__FUNCTION__) << troubleCounter << " of " << memGpuRef.size() << " objects were not released properly.\n";

            i(log).memGPU().clear();

            return (troubleCounter == 0);
        }

    private:
        static memory& i(std::ostream& log = std::cout) {
            static memory m_instance;
            m_instance.m_log = &log;
            return m_instance;
        }

        static memory& raw_i() {
            static memory m_instance;
            return m_instance;
        }

        std::ostream& log() const {
            return *m_log;
        }

        buffers_t& memGPU() {
            return m_memGPU;
        }

        buffer_t& nullBuffer() {
            return emptyBuffer;
        }

    private:
        buffers_t m_memGPU;
        buffer_t emptyBuffer = {nullptr, 0};

    private:
        memory(const memory& other) {}
        memory& operator=(const memory& other) {}
        memory(memory&& other) noexcept {}
        memory& operator=(memory&& other) noexcept {}

    }; // class memory

} // namespace ocl