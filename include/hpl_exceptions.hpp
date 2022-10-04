
#ifndef ORNL_HPL_EXCEPTIONS_H_
#define ORNL_HPL_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace ornl_hpl {

class BadAlloc : public std::runtime_error
{
public:
    BadAlloc(const std::string file, int line, const std::string device)
        : std::runtime_error("ORNL HPL: Failed alloc on " + device
                + " at " + file + std::to_string(line) + ": ")
    { }
};

#define ORNL_HPL_THROW_BAD_ALLOC(device) \
    throw ornl_hpl::BadAlloc(__FILE__, __LINE__, device)

#define ORNL_HPL_CHECK_ALLOC(_ptr, _device) \
    if(!(_ptr)) { \
        throw ornl_hpl::BadAlloc(__FILE__, __LINE__, _device); \
    } \
    static_assert(true, "Prevent error message")

class UnsupportedScalarType : public std::runtime_error
{
public:
    UnsupportedScalarType(const std::string file, int line, const std::string what)
        : std::runtime_error("ORNL HPL: Unsupported scalar type at " + file + std::to_string(line)
                + ": " + what)
    { }
};

#define ORNL_HPL_THROW_UNSUPPORTED_SCALAR_TYPE(error_msg) \
    throw ornl_hpl::UnsupportedScalarType(__FILE__, __LINE__, error_msg)

class InconsistentBlockSize : public std::runtime_error
{
public:
    InconsistentBlockSize(const std::string file, int line, const std::string what)
        : std::runtime_error("ORNL HPL: Inconsistent block size at " + file + std::to_string(line)
                + ": " + what)
    { }
};

#define ORNL_HPL_THROW_INCONSISTENT_BLOCK_SIZE(error_msg) \
    throw ornl_hpl::InconsistentBlockSize(__FILE__, __LINE__, error_msg)

class NotSupported : public std::runtime_error
{
public:
    NotSupported(const std::string file, int line, const std::string what)
        : std::runtime_error("ORNL HPL: Operation not supported at " + file + std::to_string(line)
                + ": " + what)
    { }
};

#define ORNL_HPL_THROW_NOT_SUPPORTED(unsupported_op) \
    throw ornl_hpl::NotSupported(__FILE__, __LINE__, unsupported_op)

class NotImplemented : public std::runtime_error
{
public:
    NotImplemented(const std::string file, int line, const std::string what)
        : std::runtime_error("ORNL HPL: Operation not implemented at " + file + std::to_string(line)
                + ": " + what)
    { }
};

#define ORNL_HPL_THROW_NOT_IMPLEMENTED(unimplemented_op) \
    throw ornl_hpl::NotImplemented(__FILE__, __LINE__, unimplemented_op)


class ZeroPivot : public std::runtime_error
{
public:
    ZeroPivot(const std::string file, int line, const double pivot_value, const int col)
        : std::runtime_error("ORNL HPL: ZERO pivot at " + file + std::to_string(line)
                + ": " + std::to_string(pivot_value) + " in col " + std::to_string(col))
    { }
};

#define ORNL_HPL_THROW_ZERO_PIVOT(pivot_value, col) \
    throw ornl_hpl::ZeroPivot(__FILE__, __LINE__, pivot_value, col)

}

#endif
