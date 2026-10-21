#pragma once
#include "flawn_ast.hpp"
#include <unordered_map>
#include <string>

namespace flawn {

struct AnalysisStats {
    size_t python_funcs_unboxed = 0;
    size_t flawn_funcs_compiled = 0;
    size_t simd_loops_vectorized = 0;
    size_t parallel_kernels = 0;
    size_t arena_buffers_allocated = 0;
};

class Analyzer {
public:
    AnalysisStats analyze(Program& prog);

private:
    std::unordered_map<std::string, ValueType> sym_table_;
    AnalysisStats stats_;

    void analyze_stmt(Stmt* st);
    ValueType analyze_expr(Expr* ex);
    void inspect_loop_optimizations(Stmt* loop_stmt);
};

} // namespace flawn
