#pragma once
#include "flawn_ast.hpp"
#include <string>
#include <sstream>

namespace flawn {

class NativeCodegen {
public:
    // Generates standalone C++23 + AVX2/FMA source file from analyzed Flawn AST
    std::string emit_cpp23(const Program& prog, const std::string& runtime_include_dir);

    // Invokes GCC 16 with hardware-tuned flags (-std=c++23 -O3 -march=native -mavx2 -mfma -ffast-math)
    bool compile_binary(
        const Program& prog,
        const std::string& output_exe_path,
        const std::string& project_root,
        bool verbose = false);

private:
    std::ostringstream out_;
    int indent_level_ = 0;

    void indent();
    void emit_stmt(const Stmt* st, bool inside_main = false);
    std::string emit_expr(const Expr* ex);
};

} // namespace flawn
