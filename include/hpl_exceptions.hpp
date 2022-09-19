
#ifndef ORNL_HPL_EXCEPTIONS_H_
#define ORNL_HPL_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace ornl_hpl {

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

}

#endif
