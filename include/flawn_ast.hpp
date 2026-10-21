#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace flawn {

enum class ValueType {
    Inferred, // Dynamic / To be inferred by Flawn Type Lifter
    I64,      // int64_t / Python int (unboxed)
    I32,      // int32_t
    Usize,    // size_t
    F64,      // double / Python float (unboxed)
    F32,      // float (32-bit SIMD lane)
    Bool,     // bool
    String,   // std::string / const char*
    SliceF32, // f32[] / SimdBuffer<float>
    SliceF64, // f64[] / SimdBuffer<double>
    SliceI64, // i64[] / SimdBuffer<int64_t>
    Map,      // Key-Value Dictionary / Object
    Struct,   // User-defined struct / class instance
    Void,
    PyDynamic // Boxed PyObject* fallback
};

inline std::string type_to_cpp(ValueType t, const std::string& struct_name = "") {
    switch (t) {
        case ValueType::I64:      return "int64_t";
        case ValueType::I32:      return "int32_t";
        case ValueType::Usize:    return "size_t";
        case ValueType::F64:      return "double";
        case ValueType::F32:      return "float";
        case ValueType::Bool:     return "bool";
        case ValueType::String:   return "std::string";
        case ValueType::SliceF32: return "float* FLAWN_RESTRICT";
        case ValueType::SliceF64: return "double* FLAWN_RESTRICT";
        case ValueType::SliceI64: return "int64_t* FLAWN_RESTRICT";
        case ValueType::Struct:   return struct_name.empty() ? "auto" : struct_name;
        case ValueType::Void:     return "void";
        default:                  return "double";
    }
}

struct Expr;
struct Stmt;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

enum class ExprKind {
    IntLit,
    FloatLit,
    BoolLit,
    NullLit,
    StringLit,
    FString,     // Interpolated string: f"val = {x}" or `val = ${x}`
    Var,
    Binary,
    Unary,
    Ternary,     // cond ? a : b or a if cond else b
    Call,
    MemberCall,  // e.g., math.sqrt(x), arr.sum(), str.upper()
    FieldAccess, // e.g., obj.x, vec.len
    Index,       // arr[i] or map["key"]
    ArrayAlloc,  // f32[n] or [0.0] * n
    ArrayLit,    // [1, 2, 3, 4]
    MapLit,      // { "a": 1, "b": 2 }
    StructInit   // Point { x: 1.0, y: 2.0 }
};

struct Expr {
    ExprKind kind;
    ValueType inferred_type = ValueType::Inferred;
    std::string struct_type_name;
    int line = 1;

    // Literal payloads
    int64_t int_val = 0;
    double float_val = 0.0;
    bool bool_val = false;
    std::string str_val;  // Var name, operator, function name, member/field name
    std::string obj_name; // Module/object identifier

    ExprPtr lhs;
    ExprPtr rhs;
    ExprPtr cond_expr; // For Ternary
    std::vector<ExprPtr> args;
    std::vector<std::string> keys; // For MapLit / StructInit field names, or FString text fragments
    ValueType alloc_elem_type = ValueType::F32;

    explicit Expr(ExprKind k, int ln = 1) : kind(k), line(ln) {}
};

enum class StmtKind {
    VarDecl,     // let x: i64 = 10; or Python x = 10
    Assign,      // x = expr, x += expr, arr[i] = expr, obj.field = expr
    If,          // if / elif / else (both Python & C++/JS)
    ForRange,    // Python `for i in range(start, stop, step):` or Flawn `for i in 0..n`
    ForEach,     // `for x in arr`
    ForCStyle,   // C++/JS `for (let i = 0; i < n; i++) { ... }`
    While,       // while (cond) { ... } or while cond:
    Break,       // break
    Continue,    // continue
    Assert,      // assert cond, "msg"
    Return,      // return expr
    ExprStmt,    // standalone function call / print
    Import,      // import math / from x import y
    StructDecl,  // struct Point { x: f64, y: f64 } or class Point:
    FuncDecl     // fn foo(...) -> T { ... } or def foo(...):
};

struct Param {
    std::string name;
    ValueType type = ValueType::Inferred;
    std::string struct_type_name;
    ExprPtr default_val;
};

struct StructField {
    std::string name;
    ValueType type = ValueType::F64;
    ExprPtr default_val;
};

struct Stmt {
    StmtKind kind;
    int line = 1;

    // VarDecl / Assign / Import / FuncDecl / StructDecl name
    std::string name;
    std::string field_name;      // For `obj.field = expr`
    std::string struct_type_name;// When decl_type == ValueType::Struct
    std::string assign_op = "="; // "=", "+=", "-=", "*=", "/="
    ValueType decl_type = ValueType::Inferred;
    bool is_const = false;

    // Expressions
    ExprPtr expr;       // initializer / condition / return value
    ExprPtr index_expr; // for array/map element assignment `arr[index_expr] = expr`
    ExprPtr assert_msg; // optional message for Assert

    // ForRange / ForEach / ForCStyle fields
    std::string loop_var;
    ExprPtr range_start;
    ExprPtr range_stop;
    ExprPtr range_step;
    bool range_inclusive = false;
    StmtPtr c_init;
    ExprPtr c_cond;
    StmtPtr c_step;

    // Bodies
    std::vector<StmtPtr> body;
    std::vector<StmtPtr> else_body;

    // Struct fields & methods
    std::vector<StructField> fields;

    // Function declaration metadata
    std::vector<Param> params;
    ValueType return_type = ValueType::Inferred;
    bool is_python_def = false;
    bool attr_simd = false;
    bool attr_parallel = false;
    bool attr_fastmath = false;
    bool attr_autotune = false;

    // Analyzer-computed optimization flags
    bool can_lift_simd_dot = false;
    bool can_lift_py_range_poly = false;
    std::string dot_arr_a;
    std::string dot_arr_b;
    std::string dot_len_var;

    explicit Stmt(StmtKind k, int ln = 1) : kind(k), line(ln) {}
};

struct Program {
    std::vector<std::string> imports;
    std::vector<StmtPtr> statements;
};

} // namespace flawn
