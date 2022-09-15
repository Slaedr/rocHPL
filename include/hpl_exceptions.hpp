
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

}

#endif