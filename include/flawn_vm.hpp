#pragma once
#include "flawn_ast.hpp"
#include "flawn_runtime.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace flawn {

enum class VmObjKind : uint32_t {
    String = 0x53545231, // 'STR1'
    Array  = 0x41525231, // 'ARR1'
    Map    = 0x4D415031, // 'MAP1'
    Struct = 0x53544331  // 'STC1'
};

struct VmHeader {
    VmObjKind kind;
};

struct VmString : VmHeader {
    size_t length;
    const char* c_str;
};

struct VmArray : VmHeader {
    ValueType elem_type;
    size_t length;
    size_t capacity;
    void* data;
};

struct VmMap : VmHeader {
    std::string type_name;
    std::vector<std::pair<std::string, rt::NaNVal>> entries;

    rt::NaNVal get(const std::string& k) const {
        for (const auto& kv : entries) {
            if (kv.first == k) return kv.second;
        }
        return rt::NaNVal::null();
    }
    void set(const std::string& k, rt::NaNVal v) {
        for (auto& kv : entries) {
            if (kv.first == k) {
                kv.second = v;
                return;
            }
        }
        entries.emplace_back(k, v);
    }
    bool contains(const std::string& k) const {
        for (const auto& kv : entries) {
            if (kv.first == k) return true;
        }
        return false;
    }
};

// ============================================================================
// Genuine Register-Based Bytecode ISA (Zero AST walking in hot loops)
// ============================================================================
enum class OpCode : uint8_t {
    Nop = 0,
    LoadImmI64,   // R[dst] = imm_i64
    LoadImmF64,   // R[dst] = imm_f64
    LoadConst,    // R[dst] = const_pool[imm_i64]
    Move,         // R[dst] = R[src1]

    // General Binary & Unary Operations (works for any int/float/string expression)
    Add,          // R[dst] = R[src1] + R[src2]
    Sub,          // R[dst] = R[src1] - R[src2]
    Mul,          // R[dst] = R[src1] * R[src2]
    Div,          // R[dst] = R[src1] / R[src2]
    FloorDiv,     // R[dst] = R[src1] // R[src2]
    Mod,          // R[dst] = R[src1] % R[src2]
    Pow,          // R[dst] = R[src1] ** R[src2]
    BitAnd,       // R[dst] = R[src1] & R[src2]
    BitOr,        // R[dst] = R[src1] | R[src2]
    BitXor,       // R[dst] = R[src1] ^ R[src2]
    Shl,          // R[dst] = R[src1] << R[src2]
    Shr,          // R[dst] = R[src1] >> R[src2]
    Eq,           // R[dst] = R[src1] == R[src2]
    Neq,          // R[dst] = R[src1] != R[src2]
    Lt,           // R[dst] = R[src1] < R[src2]
    Le,           // R[dst] = R[src1] <= R[src2]
    Gt,           // R[dst] = R[src1] > R[src2]
    Ge,           // R[dst] = R[src1] >= R[src2]
    And,          // R[dst] = R[src1] && R[src2]
    Or,           // R[dst] = R[src1] || R[src2]
    NullCoalesce, // R[dst] = R[src1] ?? R[src2]
    Neg,          // R[dst] = -R[src1]
    Not,          // R[dst] = !R[src1]

    // Array / Map / Struct Indexing
    GetIndex,     // R[dst] = R[src1][R[src2]]
    SetIndex,     // R[dst][R[src1]] = R[src2] (imm_i64 encodes compound op: 0:'=', 1:'+=', 2:'-=', 3:'*=')

    // Control Flow & Fast Hardware Counter Loops
    Jump,         // pc = imm_i64
    JumpIfFalse,  // if (!R[src1]) pc = imm_i64
    ForPrep,      // Initialize loop: if R[dst] >= R[src1] jump imm_i64
    ForLoopStep1, // R[dst] += 1; if (R[dst] < R[src1]) pc = imm_i64
    ForLoopStepN, // R[dst] += R[src2]; if (R[dst] < R[src1]) pc = imm_i64
    Return,       // return R[src1]

    // High-level / Complex Expr or Stmt Bridge (calls, allocations, F-strings, methods)
    EvalAstExpr,  // R[dst] = eval_ast_expr(ast_expr_pool[imm_i64], reg_base)
    ExecAstStmt   // exec_ast_stmt(ast_stmt_pool[imm_i64], reg_base)
};

struct BytecodeInst {
    OpCode op = OpCode::Nop;
    uint16_t dst = 0;
    uint16_t src1 = 0;
    uint16_t src2 = 0;
    int64_t imm_i64 = 0;
    double imm_f64 = 0.0;
};

struct CompiledFunction {
    std::string name;
    uint16_t num_regs = 0;
    uint16_t num_params = 0;
    std::vector<BytecodeInst> code;
    std::vector<const Expr*> ast_exprs;
    std::vector<const Stmt*> ast_stmts;
    std::unordered_map<std::string, uint16_t> local_reg_map;
    const Stmt* decl_stmt = nullptr;
};

class BytecodeVM {
public:
    BytecodeVM();
    void execute(const Program& prog);
    size_t arena_bytes_used() const noexcept { return arena_.total_allocated_bytes(); }
    std::string val_to_string(rt::NaNVal v) const;

private:
    rt::SlabArena arena_;
    std::vector<std::unique_ptr<VmMap>> map_pool_;
    std::unordered_map<std::string, rt::NaNVal> globals_;
    std::unordered_map<std::string, const Stmt*> functions_;
    std::unordered_map<std::string, CompiledFunction> compiled_funcs_;
    std::unordered_map<std::string, const Stmt*> structs_;

    bool has_returned_ = false;
    bool has_break_ = false;
    bool has_continue_ = false;
    rt::NaNVal return_val_ = rt::NaNVal::null();

    // Bytecode Compiler & Register Executor
    CompiledFunction compile_function(const Stmt* fn_stmt);
    rt::NaNVal run_compiled_function(const CompiledFunction& fn, const rt::NaNVal* args, size_t nargs);

    rt::NaNVal make_string(const std::string& s);
    rt::NaNVal make_map(const std::string& type_name = "");
    void exec_stmt(const Stmt* st, std::unordered_map<std::string, rt::NaNVal>& env);
    rt::NaNVal eval_expr(const Expr* ex, std::unordered_map<std::string, rt::NaNVal>& env);
    void print_val(rt::NaNVal v);
};

} // namespace flawn
