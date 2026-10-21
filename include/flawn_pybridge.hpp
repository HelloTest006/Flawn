#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "flawn_runtime.hpp"

namespace flawn::py {

class PythonBridge {
public:
    PythonBridge();
    ~PythonBridge();

    bool initialize();
    bool is_initialized() const noexcept { return initialized_; }

    // Import standard Python package (e.g., "math", "json", "statistics", or site-packages)
    bool import_module(const std::string& module_name);

    // Call a function inside an imported Python module with numeric or string args
    flawn::rt::NaNVal call_module_fn(
        const std::string& module_name,
        const std::string& fn_name,
        const std::vector<flawn::rt::NaNVal>& args);

    // Execute raw Python code or evaluate a Python expression using CPython 3.14 runtime
    bool exec_python_block(const std::string& py_code);
    std::string eval_to_string(const std::string& py_expr);
    double eval_to_double(const std::string& py_expr);

    // Set a global variable in Python's __main__ dictionary from Flawn's unboxed runtime
    void set_global_i64(const std::string& name, int64_t val);
    void set_global_f64(const std::string& name, double val);

private:
    bool initialized_ = false;
    void* lib_handle_ = nullptr;
};

// Global shared bridge instance
PythonBridge& bridge();

} // namespace flawn::py
