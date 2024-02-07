#ifndef HPL_TEST_ERROR_HANDLING_HPP_
#define HPL_TEST_ERROR_HANDLING_HPP_

#include <stdexcept>
#include <string>

namespace test {

class PreconditionNotMet : public std::runtime_error
{
public:
    PreconditionNotMet(const std::string& msg)
        : std::runtime_error("Test precondition not met: " + msg)
        { }
};

class TestFailed : public std::runtime_error
{
public:
    TestFailed(const std::string& msg)
        : std::runtime_error("Test failed: " + msg)
        { }
};

}


#endif // HPL_TEST_ERROR_HANDLING_HPP_
