#include "flawn_vm.hpp"
#include "flawn_pybridge.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace flawn {

BytecodeVM::BytecodeVM() : arena_(32 * 1024 * 1024) {}

rt::NaNVal BytecodeVM::make_string(const std::string& s) {
    VmString* vs = static_cast<VmString*>(arena_.alloc(sizeof(VmString), 16));
    vs->kind = VmObjKind::String;
    vs->length = s.size();
    char* buf = arena_.alloc_array<char>(s.size() + 1);
    std::memcpy(buf, s.c_str(), s.size() + 1);
    vs->c_str = buf;
    return rt::NaNVal::from_ptr(vs);
}

rt::NaNVal BytecodeVM::make_map(const std::string& type_name) {
    auto m = std::make_unique<VmMap>();
    m->kind = type_name.empty() ? VmObjKind::Map : VmObjKind::Struct;
    m->type_name = type_name;
    VmMap* ptr = m.get();
    map_pool_.push_back(std::move(m));
    return rt::NaNVal::from_ptr(ptr);
}

std::string BytecodeVM::val_to_string(rt::NaNVal v) const {
    if (v.is_int()) {
        return std::to_string(v.as_i64());
    }
    if (v.is_double()) {
        std::ostringstream ss;
        double d = v.as_double();
        if (std::floor(d) == d && std::abs(d) < 1e15) {
            ss << std::fixed << std::setprecision(4) << d;
            std::string s = ss.str();
            while (s.size() > 2 && s.back() == '0' && s[s.size() - 2] != '.') s.pop_back();
            return s;
        }
        ss << d;
        return ss.str();
    }
    if (v.is_bool()) {
        return v.as_i64() ? "true" : "false";
    }
    if (v.is_ptr()) {
        const VmHeader* hdr = static_cast<const VmHeader*>(v.as_ptr());
        if (!hdr) return "null";
        if (hdr->kind == VmObjKind::String) {
            const VmString* vs = static_cast<const VmString*>(hdr);
            return vs->c_str ? std::string(vs->c_str, vs->length) : "";
        }
        if (hdr->kind == VmObjKind::Array) {
            const VmArray* arr = static_cast<const VmArray*>(hdr);
            std::ostringstream ss;
            ss << "[";
            size_t show = std::min<size_t>(arr->length, 16);
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) ss << ", ";
                if (arr->elem_type == ValueType::F32) ss << static_cast<const float*>(arr->data)[i];
                else if (arr->elem_type == ValueType::F64) ss << static_cast<const double*>(arr->data)[i];
                else ss << static_cast<const int64_t*>(arr->data)[i];
            }
            if (arr->length > show) ss << ", ... (" << arr->length << " total)";
            ss << "]";
            return ss.str();
        }
        if (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct) {
            const VmMap* mp = static_cast<const VmMap*>(hdr);
            std::ostringstream ss;
            if (!mp->type_name.empty()) ss << mp->type_name << " ";
            ss << "{";
            for (size_t i = 0; i < mp->entries.size(); ++i) {
                if (i > 0) ss << ", ";
                if (mp->type_name.empty()) ss << "\"" << mp->entries[i].first << "\": ";
                else ss << mp->entries[i].first << ": ";
                if (mp->entries[i].second.is_ptr() &&
                    static_cast<const VmHeader*>(mp->entries[i].second.as_ptr())->kind == VmObjKind::String) {
                    ss << "\"" << val_to_string(mp->entries[i].second) << "\"";
                } else {
                    ss << val_to_string(mp->entries[i].second);
                }
            }
            ss << "}";
            return ss.str();
        }
    }
    return "null";
}

// ============================================================================
// Register-Based Bytecode Compiler
// Compiles arbitrary functions, arithmetic expressions, loops, and branches
// into a flat register instruction stream (zero hash-table lookups in loops).
// ============================================================================
struct FunctionCompiler {
    CompiledFunction out;
    std::vector<std::vector<size_t>> break_Fixups;
    std::vector<std::vector<size_t>> continue_fixups;

    uint16_t get_or_alloc_local(const std::string& name) {
        auto it = out.local_reg_map.find(name);
        if (it != out.local_reg_map.end()) return it->second;
        uint16_t r = out.num_regs++;
        out.local_reg_map[name] = r;
        return r;
    }

    uint16_t alloc_temp() {
        return out.num_regs++;
    }

    bool can_compile_expr_pure(const Expr* ex) const {
        if (!ex) return true;
        switch (ex->kind) {
            case ExprKind::IntLit:
            case ExprKind::FloatLit:
            case ExprKind::BoolLit:
            case ExprKind::NullLit:
                return true;
            case ExprKind::Var:
                return out.local_reg_map.find(ex->str_val) != out.local_reg_map.end();
            case ExprKind::Unary:
                return can_compile_expr_pure(ex->lhs.get());
            case ExprKind::Binary:
                return can_compile_expr_pure(ex->lhs.get()) && can_compile_expr_pure(ex->rhs.get());
            case ExprKind::Ternary:
                return can_compile_expr_pure(ex->cond_expr.get()) &&
                       can_compile_expr_pure(ex->lhs.get()) &&
                       can_compile_expr_pure(ex->rhs.get());
            case ExprKind::Index:
                return can_compile_expr_pure(ex->lhs.get()) && can_compile_expr_pure(ex->rhs.get());
            default:
                return false;
        }
    }

    uint16_t compile_expr(const Expr* ex, int preferred_dst = -1) {
        uint16_t dst = (preferred_dst >= 0) ? static_cast<uint16_t>(preferred_dst) : alloc_temp();
        if (!ex) {
            out.code.push_back(BytecodeInst{OpCode::LoadImmI64, dst, 0, 0, 0, 0.0});
            return dst;
        }

        if (!can_compile_expr_pure(ex)) {
            int64_t idx = static_cast<int64_t>(out.ast_exprs.size());
            out.ast_exprs.push_back(ex);
            out.code.push_back(BytecodeInst{OpCode::EvalAstExpr, dst, 0, 0, idx, 0.0});
            return dst;
        }

        switch (ex->kind) {
            case ExprKind::IntLit:
                out.code.push_back(BytecodeInst{OpCode::LoadImmI64, dst, 0, 0, ex->int_val, 0.0});
                return dst;
            case ExprKind::FloatLit:
                out.code.push_back(BytecodeInst{OpCode::LoadImmF64, dst, 0, 0, 0, ex->float_val});
                return dst;
            case ExprKind::BoolLit:
                out.code.push_back(BytecodeInst{OpCode::LoadImmI64, dst, 0, 0, ex->bool_val ? 1 : 0, 0.0});
                return dst;
            case ExprKind::NullLit:
                out.code.push_back(BytecodeInst{OpCode::LoadImmI64, dst, 0, 0, 0, 0.0});
                return dst;
            case ExprKind::Var: {
                uint16_t vreg = out.local_reg_map.at(ex->str_val);
                if (preferred_dst >= 0 && static_cast<uint16_t>(preferred_dst) != vreg) {
                    out.code.push_back(BytecodeInst{OpCode::Move, dst, vreg, 0, 0, 0.0});
                    return dst;
                }
                return vreg;
            }
            case ExprKind::Unary: {
                uint16_t s1 = compile_expr(ex->lhs.get());
                OpCode op = (ex->str_val == "-") ? OpCode::Neg : OpCode::Not;
                out.code.push_back(BytecodeInst{op, dst, s1, 0, 0, 0.0});
                return dst;
            }
            case ExprKind::Index: {
                uint16_t arr_r = compile_expr(ex->lhs.get());
                uint16_t idx_r = compile_expr(ex->rhs.get());
                out.code.push_back(BytecodeInst{OpCode::GetIndex, dst, arr_r, idx_r, 0, 0.0});
                return dst;
            }
            case ExprKind::Ternary: {
                uint16_t cond_r = compile_expr(ex->cond_expr.get());
                size_t jfalse_idx = out.code.size();
                out.code.push_back(BytecodeInst{OpCode::JumpIfFalse, 0, cond_r, 0, 0, 0.0});
                compile_expr(ex->lhs.get(), dst);
                size_t jend_idx = out.code.size();
                out.code.push_back(BytecodeInst{OpCode::Jump, 0, 0, 0, 0, 0.0});
                out.code[jfalse_idx].imm_i64 = static_cast<int64_t>(out.code.size());
                compile_expr(ex->rhs.get(), dst);
                out.code[jend_idx].imm_i64 = static_cast<int64_t>(out.code.size());
                return dst;
            }
            case ExprKind::Binary: {
                uint16_t l_r = compile_expr(ex->lhs.get());
                uint16_t r_r = compile_expr(ex->rhs.get());
                const std::string& s = ex->str_val;
                OpCode op = OpCode::Add;
                if (s == "+") op = OpCode::Add;
                else if (s == "-") op = OpCode::Sub;
                else if (s == "*") op = OpCode::Mul;
                else if (s == "/") op = OpCode::Div;
                else if (s == "//") op = OpCode::FloorDiv;
                else if (s == "%") op = OpCode::Mod;
                else if (s == "**") op = OpCode::Pow;
                else if (s == "&") op = OpCode::BitAnd;
                else if (s == "|") op = OpCode::BitOr;
                else if (s == "^") op = OpCode::BitXor;
                else if (s == "<<") op = OpCode::Shl;
                else if (s == ">>") op = OpCode::Shr;
                else if (s == "==") op = OpCode::Eq;
                else if (s == "!=") op = OpCode::Neq;
                else if (s == "<") op = OpCode::Lt;
                else if (s == "<=") op = OpCode::Le;
                else if (s == ">") op = OpCode::Gt;
                else if (s == ">=") op = OpCode::Ge;
                else if (s == "&&") op = OpCode::And;
                else if (s == "||") op = OpCode::Or;
                else if (s == "??") op = OpCode::NullCoalesce;
                out.code.push_back(BytecodeInst{op, dst, l_r, r_r, 0, 0.0});
                return dst;
            }
            default:
                return dst;
        }
    }

    void PreScanLocals(const std::vector<StmtPtr>& stmts) {
        for (const auto& s : stmts) {
            if (!s) continue;
            if (s->kind == StmtKind::VarDecl || (s->kind == StmtKind::Assign && !s->index_expr && s->field_name.empty())) {
                get_or_alloc_local(s->name);
            } else if (s->kind == StmtKind::ForRange || s->kind == StmtKind::ForEach) {
                get_or_alloc_local(s->loop_var);
                PreScanLocals(s->body);
            } else if (s->kind == StmtKind::ForCStyle) {
                if (s->c_init && (s->c_init->kind == StmtKind::VarDecl || s->c_init->kind == StmtKind::Assign)) {
                    get_or_alloc_local(s->c_init->name);
                }
                PreScanLocals(s->body);
            } else if (s->kind == StmtKind::If) {
                PreScanLocals(s->body);
                PreScanLocals(s->else_body);
            } else if (s->kind == StmtKind::While) {
                PreScanLocals(s->body);
            }
        }
    }

    void compile_stmt(const Stmt* st) {
        if (!st) return;

        switch (st->kind) {
            case StmtKind::VarDecl: {
                uint16_t dst = get_or_alloc_local(st->name);
                if (st->expr) {
                    compile_expr(st->expr.get(), dst);
                } else {
                    out.code.push_back(BytecodeInst{OpCode::LoadImmI64, dst, 0, 0, 0, 0.0});
                }
                break;
            }
            case StmtKind::Assign: {
                if (!st->field_name.empty()) {
                    int64_t idx = static_cast<int64_t>(out.ast_stmts.size());
                    out.ast_stmts.push_back(st);
                    out.code.push_back(BytecodeInst{OpCode::ExecAstStmt, 0, 0, 0, idx, 0.0});
                    break;
                }
                if (st->index_expr) {
                    uint16_t arr_r = get_or_alloc_local(st->name);
                    uint16_t idx_r = compile_expr(st->index_expr.get());
                    uint16_t val_r = compile_expr(st->expr.get());
                    int64_t mode = 0;
                    if (st->assign_op == "+=") mode = 1;
                    else if (st->assign_op == "-=") mode = 2;
                    else if (st->assign_op == "*=") mode = 3;
                    out.code.push_back(BytecodeInst{OpCode::SetIndex, arr_r, idx_r, val_r, mode, 0.0});
                    break;
                }
                uint16_t dst = get_or_alloc_local(st->name);
                if (st->assign_op == "=") {
                    compile_expr(st->expr.get(), dst);
                } else {
                    uint16_t rhs_r = compile_expr(st->expr.get());
                    OpCode op = OpCode::Add;
                    if (st->assign_op == "+=") op = OpCode::Add;
                    else if (st->assign_op == "-=") op = OpCode::Sub;
                    else if (st->assign_op == "*=") op = OpCode::Mul;
                    else if (st->assign_op == "/=") op = OpCode::Div;
                    out.code.push_back(BytecodeInst{op, dst, dst, rhs_r, 0, 0.0});
                }
                break;
            }
            case StmtKind::If: {
                uint16_t cond_r = compile_expr(st->expr.get());
                size_t jfalse_idx = out.code.size();
                out.code.push_back(BytecodeInst{OpCode::JumpIfFalse, 0, cond_r, 0, 0, 0.0});
                for (const auto& b : st->body) compile_stmt(b.get());
                if (!st->else_body.empty()) {
                    size_t jend_idx = out.code.size();
                    out.code.push_back(BytecodeInst{OpCode::Jump, 0, 0, 0, 0, 0.0});
                    out.code[jfalse_idx].imm_i64 = static_cast<int64_t>(out.code.size());
                    for (const auto& eb : st->else_body) compile_stmt(eb.get());
                    out.code[jend_idx].imm_i64 = static_cast<int64_t>(out.code.size());
                } else {
                    out.code[jfalse_idx].imm_i64 = static_cast<int64_t>(out.code.size());
                }
                break;
            }
            case StmtKind::ForRange: {
                uint16_t ivar = get_or_alloc_local(st->loop_var);
                compile_expr(st->range_start.get(), ivar);
                uint16_t stop_r = compile_expr(st->range_stop.get());
                bool step_is_one = (st->range_step && st->range_step->kind == ExprKind::IntLit && st->range_step->int_val == 1);
                uint16_t step_r = step_is_one ? 0 : compile_expr(st->range_step.get());

                size_t prep_idx = out.code.size();
                out.code.push_back(BytecodeInst{OpCode::ForPrep, ivar, stop_r, 0, 0, 0.0});
                size_t body_start = out.code.size();

                break_Fixups.emplace_back();
                continue_fixups.emplace_back();
                for (const auto& b : st->body) compile_stmt(b.get());

                size_t step_pc = out.code.size();
                for (size_t c_idx : continue_fixups.back()) {
                    out.code[c_idx].imm_i64 = static_cast<int64_t>(step_pc);
                }
                if (step_is_one) {
                    out.code.push_back(BytecodeInst{OpCode::ForLoopStep1, ivar, stop_r, 0, static_cast<int64_t>(body_start), 0.0});
                } else {
                    out.code.push_back(BytecodeInst{OpCode::ForLoopStepN, ivar, stop_r, step_r, static_cast<int64_t>(body_start), 0.0});
                }
                size_t loop_exit = out.code.size();
                out.code[prep_idx].imm_i64 = static_cast<int64_t>(loop_exit);
                for (size_t b_idx : break_Fixups.back()) {
                    out.code[b_idx].imm_i64 = static_cast<int64_t>(loop_exit);
                }
                break_Fixups.pop_back();
                continue_fixups.pop_back();
                break;
            }
            case StmtKind::ForCStyle: {
                compile_stmt(st->c_init.get());
                size_t cond_pc = out.code.size();
                uint16_t cond_r = compile_expr(st->c_cond.get());
                size_t jexit_idx = out.code.size();
                out.code.push_back(BytecodeInst{OpCode::JumpIfFalse, 0, cond_r, 0, 0, 0.0});

                break_Fixups.emplace_back();
                continue_fixups.emplace_back();
                for (const auto& b : st->body) compile_stmt(b.get());

                size_t step_pc = out.code.size();
                for (size_t c_idx : continue_fixups.back()) {
                    out.code[c_idx].imm_i64 = static_cast<int64_t>(step_pc);
                }
                compile_stmt(st->c_step.get());
                out.code.push_back(BytecodeInst{OpCode::Jump, 0, 0, 0, static_cast<int64_t>(cond_pc), 0.0});

                size_t loop_exit = out.code.size();
                out.code[jexit_idx].imm_i64 = static_cast<int64_t>(loop_exit);
                for (size_t b_idx : break_Fixups.back()) {
                    out.code[b_idx].imm_i64 = static_cast<int64_t>(loop_exit);
                }
                break_Fixups.pop_back();
                continue_fixups.pop_back();
                break;
            }
            case StmtKind::While: {
                size_t cond_pc = out.code.size();
                uint16_t cond_r = compile_expr(st->expr.get());
                size_t jexit_idx = out.code.size();
                out.code.push_back(BytecodeInst{OpCode::JumpIfFalse, 0, cond_r, 0, 0, 0.0});

                break_Fixups.emplace_back();
                continue_fixups.emplace_back();
                for (const auto& b : st->body) compile_stmt(b.get());

                for (size_t c_idx : continue_fixups.back()) {
                    out.code[c_idx].imm_i64 = static_cast<int64_t>(cond_pc);
                }
                out.code.push_back(BytecodeInst{OpCode::Jump, 0, 0, 0, static_cast<int64_t>(cond_pc), 0.0});
                size_t loop_exit = out.code.size();
                out.code[jexit_idx].imm_i64 = static_cast<int64_t>(loop_exit);
                for (size_t b_idx : break_Fixups.back()) {
                    out.code[b_idx].imm_i64 = static_cast<int64_t>(loop_exit);
                }
                break_Fixups.pop_back();
                continue_fixups.pop_back();
                break;
            }
            case StmtKind::Break: {
                if (!break_Fixups.empty()) {
                    break_Fixups.back().push_back(out.code.size());
                    out.code.push_back(BytecodeInst{OpCode::Jump, 0, 0, 0, 0, 0.0});
                }
                break;
            }
            case StmtKind::Continue: {
                if (!continue_fixups.empty()) {
                    continue_fixups.back().push_back(out.code.size());
                    out.code.push_back(BytecodeInst{OpCode::Jump, 0, 0, 0, 0, 0.0});
                }
                break;
            }
            case StmtKind::Return: {
                uint16_t r = st->expr ? compile_expr(st->expr.get()) : alloc_temp();
                if (!st->expr) {
                    out.code.push_back(BytecodeInst{OpCode::LoadImmI64, r, 0, 0, 0, 0.0});
                }
                out.code.push_back(BytecodeInst{OpCode::Return, 0, r, 0, 0, 0.0});
                break;
            }
            default: {
                int64_t idx = static_cast<int64_t>(out.ast_stmts.size());
                out.ast_stmts.push_back(st);
                out.code.push_back(BytecodeInst{OpCode::ExecAstStmt, 0, 0, 0, idx, 0.0});
                break;
            }
        }
    }
};

CompiledFunction BytecodeVM::compile_function(const Stmt* fn_stmt) {
    FunctionCompiler fc;
    fc.out.name = fn_stmt->name;
    fc.out.decl_stmt = fn_stmt;
    fc.out.num_params = static_cast<uint16_t>(fn_stmt->params.size());
    for (const auto& p : fn_stmt->params) {
        fc.get_or_alloc_local(p.name);
    }
    fc.PreScanLocals(fn_stmt->body);
    for (const auto& b : fn_stmt->body) {
        fc.compile_stmt(b.get());
    }
    uint16_t z = fc.alloc_temp();
    fc.out.code.push_back(BytecodeInst{OpCode::LoadImmI64, z, 0, 0, 0, 0.0});
    fc.out.code.push_back(BytecodeInst{OpCode::Return, 0, z, 0, 0, 0.0});
    return fc.out;
}

// ============================================================================
// Tight Register-Based Bytecode Dispatch Loop
// Executes any general arithmetic/bitwise/branch/array loop in contiguous
// 64-bit NaN-boxed registers with zero hash-table overhead.
// ============================================================================
FLAWN_HOT rt::NaNVal BytecodeVM::run_compiled_function(
    const CompiledFunction& fn,
    const rt::NaNVal* args,
    size_t nargs)
{
    std::vector<rt::NaNVal> regs(fn.num_regs + 4, rt::NaNVal::from_i64(0));
    rt::NaNVal* FLAWN_RESTRICT R = regs.data();

    for (size_t i = 0; i < fn.num_params; ++i) {
        if (i < nargs) {
            R[i] = args[i];
        } else if (fn.decl_stmt && fn.decl_stmt->params[i].default_val) {
            std::unordered_map<std::string, rt::NaNVal> dummy;
            R[i] = eval_expr(fn.decl_stmt->params[i].default_val.get(), dummy);
        }
    }

    const BytecodeInst* FLAWN_RESTRICT code = fn.code.data();
    size_t pc = 0;
    const size_t code_len = fn.code.size();

    while (pc < code_len) {
        const BytecodeInst& ins = code[pc++];
        switch (ins.op) {
            case OpCode::Nop:
                break;
            case OpCode::LoadImmI64:
                R[ins.dst] = rt::NaNVal::from_i64(ins.imm_i64);
                break;
            case OpCode::LoadImmF64:
                R[ins.dst] = rt::NaNVal::from_double(ins.imm_f64);
                break;
            case OpCode::Move:
                R[ins.dst] = R[ins.src1];
                break;
            case OpCode::Add: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_i64(a.as_i64() + b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_double(a.to_number() + b.to_number());
                break;
            }
            case OpCode::Sub: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_i64(a.as_i64() - b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_double(a.to_number() - b.to_number());
                break;
            }
            case OpCode::Mul: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_i64(a.as_i64() * b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_double(a.to_number() * b.to_number());
                break;
            }
            case OpCode::Div: {
                R[ins.dst] = rt::NaNVal::from_double(R[ins.src1].to_number() / R[ins.src2].to_number());
                break;
            }
            case OpCode::FloorDiv: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) {
                    int64_t d = b.as_i64();
                    R[ins.dst] = rt::NaNVal::from_i64(d != 0 ? (a.as_i64() / d) : 0);
                } else {
                    R[ins.dst] = rt::NaNVal::from_i64(static_cast<int64_t>(std::floor(a.to_number() / b.to_number())));
                }
                break;
            }
            case OpCode::Mod: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) {
                    int64_t d = b.as_i64();
                    R[ins.dst] = rt::NaNVal::from_i64(d != 0 ? (a.as_i64() % d) : 0);
                } else {
                    R[ins.dst] = rt::NaNVal::from_double(std::fmod(a.to_number(), b.to_number()));
                }
                break;
            }
            case OpCode::Pow: {
                R[ins.dst] = rt::NaNVal::from_double(std::pow(R[ins.src1].to_number(), R[ins.src2].to_number()));
                break;
            }
            case OpCode::BitAnd:
                R[ins.dst] = rt::NaNVal::from_i64(R[ins.src1].as_i64() & R[ins.src2].as_i64());
                break;
            case OpCode::BitOr:
                R[ins.dst] = rt::NaNVal::from_i64(R[ins.src1].as_i64() | R[ins.src2].as_i64());
                break;
            case OpCode::BitXor:
                R[ins.dst] = rt::NaNVal::from_i64(R[ins.src1].as_i64() ^ R[ins.src2].as_i64());
                break;
            case OpCode::Shl:
                R[ins.dst] = rt::NaNVal::from_i64(R[ins.src1].as_i64() << R[ins.src2].as_i64());
                break;
            case OpCode::Shr:
                R[ins.dst] = rt::NaNVal::from_i64(R[ins.src1].as_i64() >> R[ins.src2].as_i64());
                break;
            case OpCode::Eq: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_bool(a.as_i64() == b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_bool(a.to_number() == b.to_number());
                break;
            }
            case OpCode::Neq: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_bool(a.as_i64() != b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_bool(a.to_number() != b.to_number());
                break;
            }
            case OpCode::Lt: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_bool(a.as_i64() < b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_bool(a.to_number() < b.to_number());
                break;
            }
            case OpCode::Le: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_bool(a.as_i64() <= b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_bool(a.to_number() <= b.to_number());
                break;
            }
            case OpCode::Gt: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_bool(a.as_i64() > b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_bool(a.to_number() > b.to_number());
                break;
            }
            case OpCode::Ge: {
                rt::NaNVal a = R[ins.src1], b = R[ins.src2];
                if (a.is_int() && b.is_int()) R[ins.dst] = rt::NaNVal::from_bool(a.as_i64() >= b.as_i64());
                else R[ins.dst] = rt::NaNVal::from_bool(a.to_number() >= b.to_number());
                break;
            }
            case OpCode::And: {
                bool lb = R[ins.src1].is_bool() ? (R[ins.src1].as_i64() != 0) : (R[ins.src1].to_number() != 0.0);
                bool rb = R[ins.src2].is_bool() ? (R[ins.src2].as_i64() != 0) : (R[ins.src2].to_number() != 0.0);
                R[ins.dst] = rt::NaNVal::from_bool(lb && rb);
                break;
            }
            case OpCode::Or: {
                bool lb = R[ins.src1].is_bool() ? (R[ins.src1].as_i64() != 0) : (R[ins.src1].to_number() != 0.0);
                bool rb = R[ins.src2].is_bool() ? (R[ins.src2].as_i64() != 0) : (R[ins.src2].to_number() != 0.0);
                R[ins.dst] = rt::NaNVal::from_bool(lb || rb);
                break;
            }
            case OpCode::NullCoalesce: {
                rt::NaNVal l = R[ins.src1];
                R[ins.dst] = ((l.bits & rt::NaNVal::TAG_MASK) != rt::NaNVal::TAG_NULL) ? l : R[ins.src2];
                break;
            }
            case OpCode::Neg: {
                rt::NaNVal v = R[ins.src1];
                R[ins.dst] = v.is_int() ? rt::NaNVal::from_i64(-v.as_i64()) : rt::NaNVal::from_double(-v.as_double());
                break;
            }
            case OpCode::Not: {
                rt::NaNVal v = R[ins.src1];
                bool b = v.is_bool() ? (v.as_i64() != 0) : (v.to_number() != 0.0);
                R[ins.dst] = rt::NaNVal::from_bool(!b);
                break;
            }
            case OpCode::GetIndex: {
                rt::NaNVal arr_v = R[ins.src1];
                rt::NaNVal idx_v = R[ins.src2];
                if (arr_v.is_ptr() && arr_v.as_ptr()) {
                    VmHeader* hdr = static_cast<VmHeader*>(arr_v.as_ptr());
                    if (hdr->kind == VmObjKind::Array) {
                        VmArray* arr = static_cast<VmArray*>(hdr);
                        int64_t raw_idx = idx_v.is_int() ? idx_v.as_i64() : static_cast<int64_t>(idx_v.as_double());
                        if (raw_idx < 0) raw_idx += static_cast<int64_t>(arr->length);
                        size_t idx = static_cast<size_t>(raw_idx);
                        if (idx < arr->length) {
                            if (arr->elem_type == ValueType::F32) R[ins.dst] = rt::NaNVal::from_double(static_cast<float*>(arr->data)[idx]);
                            else if (arr->elem_type == ValueType::F64) R[ins.dst] = rt::NaNVal::from_double(static_cast<double*>(arr->data)[idx]);
                            else R[ins.dst] = rt::NaNVal::from_i64(static_cast<int64_t*>(arr->data)[idx]);
                        }
                    } else if (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct) {
                        R[ins.dst] = static_cast<VmMap*>(hdr)->get(val_to_string(idx_v));
                    }
                }
                break;
            }
            case OpCode::SetIndex: {
                rt::NaNVal arr_v = R[ins.dst];
                rt::NaNVal idx_v = R[ins.src1];
                rt::NaNVal rhs   = R[ins.src2];
                if (arr_v.is_ptr() && arr_v.as_ptr()) {
                    VmHeader* hdr = static_cast<VmHeader*>(arr_v.as_ptr());
                    if (hdr->kind == VmObjKind::Array) {
                        VmArray* arr = static_cast<VmArray*>(hdr);
                        int64_t raw_idx = idx_v.is_int() ? idx_v.as_i64() : static_cast<int64_t>(idx_v.as_double());
                        if (raw_idx < 0) raw_idx += static_cast<int64_t>(arr->length);
                        size_t idx = static_cast<size_t>(raw_idx);
                        if (idx < arr->length) {
                            if (arr->elem_type == ValueType::F32) {
                                float* p = static_cast<float*>(arr->data);
                                float fv = static_cast<float>(rhs.to_number());
                                if (ins.imm_i64 == 0) p[idx] = fv;
                                else if (ins.imm_i64 == 1) p[idx] += fv;
                                else if (ins.imm_i64 == 2) p[idx] -= fv;
                                else if (ins.imm_i64 == 3) p[idx] *= fv;
                            } else if (arr->elem_type == ValueType::F64) {
                                double* p = static_cast<double*>(arr->data);
                                double dv = rhs.to_number();
                                if (ins.imm_i64 == 0) p[idx] = dv;
                                else if (ins.imm_i64 == 1) p[idx] += dv;
                                else if (ins.imm_i64 == 2) p[idx] -= dv;
                                else if (ins.imm_i64 == 3) p[idx] *= dv;
                            } else {
                                int64_t* p = static_cast<int64_t*>(arr->data);
                                int64_t iv = rhs.is_int() ? rhs.as_i64() : static_cast<int64_t>(rhs.as_double());
                                if (ins.imm_i64 == 0) p[idx] = iv;
                                else if (ins.imm_i64 == 1) p[idx] += iv;
                                else if (ins.imm_i64 == 2) p[idx] -= iv;
                                else if (ins.imm_i64 == 3) p[idx] *= iv;
                            }
                        }
                    } else if (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct) {
                        static_cast<VmMap*>(hdr)->set(val_to_string(idx_v), rhs);
                    }
                }
                break;
            }
            case OpCode::Jump:
                pc = static_cast<size_t>(ins.imm_i64);
                break;
            case OpCode::JumpIfFalse: {
                rt::NaNVal c = R[ins.src1];
                bool cond = c.is_bool() ? (c.as_i64() != 0) : (c.to_number() != 0.0);
                if (!cond) pc = static_cast<size_t>(ins.imm_i64);
                break;
            }
            case OpCode::ForPrep: {
                int64_t cur = R[ins.dst].is_int() ? R[ins.dst].as_i64() : static_cast<int64_t>(R[ins.dst].to_number());
                int64_t lim = R[ins.src1].is_int() ? R[ins.src1].as_i64() : static_cast<int64_t>(R[ins.src1].to_number());
                R[ins.dst] = rt::NaNVal::from_i64(cur);
                if (cur >= lim) pc = static_cast<size_t>(ins.imm_i64);
                break;
            }
            case OpCode::ForLoopStep1: {
                int64_t next = R[ins.dst].as_i64() + 1;
                int64_t lim  = R[ins.src1].is_int() ? R[ins.src1].as_i64() : static_cast<int64_t>(R[ins.src1].to_number());
                R[ins.dst] = rt::NaNVal::from_i64(next);
                if (next < lim) pc = static_cast<size_t>(ins.imm_i64);
                break;
            }
            case OpCode::ForLoopStepN: {
                int64_t step = R[ins.src2].as_i64();
                int64_t next = R[ins.dst].as_i64() + step;
                int64_t lim  = R[ins.src1].is_int() ? R[ins.src1].as_i64() : static_cast<int64_t>(R[ins.src1].to_number());
                R[ins.dst] = rt::NaNVal::from_i64(next);
                if (step > 0 ? (next < lim) : (next > lim)) pc = static_cast<size_t>(ins.imm_i64);
                break;
            }
            case OpCode::Return:
                return R[ins.src1];
            case OpCode::EvalAstExpr: {
                std::unordered_map<std::string, rt::NaNVal> sync_env;
                for (const auto& kv : fn.local_reg_map) sync_env[kv.first] = R[kv.second];
                R[ins.dst] = eval_expr(fn.ast_exprs[static_cast<size_t>(ins.imm_i64)], sync_env);
                for (const auto& kv : fn.local_reg_map) R[kv.second] = sync_env[kv.first];
                break;
            }
            case OpCode::ExecAstStmt: {
                std::unordered_map<std::string, rt::NaNVal> sync_env;
                for (const auto& kv : fn.local_reg_map) sync_env[kv.first] = R[kv.second];
                exec_stmt(fn.ast_stmts[static_cast<size_t>(ins.imm_i64)], sync_env);
                for (const auto& kv : fn.local_reg_map) R[kv.second] = sync_env[kv.first];
                if (has_returned_) {
                    has_returned_ = false;
                    return return_val_;
                }
                break;
            }
            default:
                break;
        }
    }
    return rt::NaNVal::null();
}

void BytecodeVM::execute(const Program& prog) {
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& imp : prog.imports) {
        if (imp != "math" && imp != "fs" && imp != "json" && imp != "http" && imp != "env" && imp != "crypto" && imp != "simd") {
            py::bridge().import_module(imp);
        }
    }
    for (const auto& st : prog.statements) {
        if (!st) continue;
        if (st->kind == StmtKind::FuncDecl) {
            functions_[st->name] = st.get();
            if (st->name != "main") {
                compiled_funcs_[st->name] = compile_function(st.get());
            }
        } else if (st->kind == StmtKind::StructDecl) {
            structs_[st->name] = st.get();
        }
    }
    for (const auto& st : prog.statements) {
        if (st && st->kind != StmtKind::FuncDecl && st->kind != StmtKind::StructDecl) {
            exec_stmt(st.get(), globals_);
        }
    }
    auto it = functions_.find("main");
    if (it != functions_.end()) {
        bool main_called = false;
        for (const auto& st : prog.statements) {
            if (st && st->kind == StmtKind::ExprStmt && st->expr &&
                st->expr->kind == ExprKind::Call && st->expr->str_val == "main") {
                main_called = true;
                break;
            }
        }
        if (!main_called) {
            std::unordered_map<std::string, rt::NaNVal> local_env;
            has_returned_ = false;
            for (const auto& b : it->second->body) {
                exec_stmt(b.get(), local_env);
                if (has_returned_) break;
            }
        }
    }
}

void BytecodeVM::exec_stmt(const Stmt* st, std::unordered_map<std::string, rt::NaNVal>& env) {
    if (!st || has_returned_ || has_break_ || has_continue_) return;

    switch (st->kind) {
        case StmtKind::VarDecl: {
            rt::NaNVal v = st->expr ? eval_expr(st->expr.get(), env) : rt::NaNVal::from_i64(0);
            env[st->name] = v;
            break;
        }
        case StmtKind::Break: {
            has_break_ = true;
            break;
        }
        case StmtKind::Continue: {
            has_continue_ = true;
            break;
        }
        case StmtKind::Assert: {
            rt::NaNVal c = eval_expr(st->expr.get(), env);
            bool ok = c.is_bool() ? (c.as_i64() != 0) : (c.to_number() != 0.0);
            if (!ok) {
                std::string msg = st->assert_msg ? val_to_string(eval_expr(st->assert_msg.get(), env)) : "Assertion failed";
                std::cerr << "[Flawn Assertion Error] Line " << st->line << ": " << msg << "\n";
                std::exit(1);
            }
            break;
        }
        case StmtKind::Assign: {
            rt::NaNVal rhs = eval_expr(st->expr.get(), env);

            // Struct / Map field assignment `obj.field = val`
            if (!st->field_name.empty()) {
                auto it = env.find(st->name);
                if (it == env.end()) it = globals_.find(st->name);
                if (it != env.end() && it != globals_.end() ? true : (it != globals_.end())) {
                    if (it->second.is_ptr()) {
                        VmHeader* hdr = static_cast<VmHeader*>(it->second.as_ptr());
                        if (hdr && (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct)) {
                            VmMap* mp = static_cast<VmMap*>(hdr);
                            if (st->assign_op == "=") {
                                mp->set(st->field_name, rhs);
                            } else {
                                rt::NaNVal cur = mp->get(st->field_name);
                                double a = cur.to_number(), b = rhs.to_number();
                                if (st->assign_op == "+=") mp->set(st->field_name, rt::NaNVal::from_double(a + b));
                                else if (st->assign_op == "-=") mp->set(st->field_name, rt::NaNVal::from_double(a - b));
                                else if (st->assign_op == "*=") mp->set(st->field_name, rt::NaNVal::from_double(a * b));
                            }
                        }
                    }
                }
                break;
            }

            // Array or Map indexed assignment `arr[i] = val` or `map["k"] = val`
            if (st->index_expr) {
                rt::NaNVal idx_v = eval_expr(st->index_expr.get(), env);
                auto it = env.find(st->name);
                if (it == env.end()) it = globals_.find(st->name);
                if (it != env.end() && it != globals_.end() ? true : (it != globals_.end())) {
                    if (!it->second.is_ptr()) break;
                    VmHeader* hdr = static_cast<VmHeader*>(it->second.as_ptr());
                    if (hdr && (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct)) {
                        VmMap* mp = static_cast<VmMap*>(hdr);
                        mp->set(val_to_string(idx_v), rhs);
                        break;
                    }
                    if (hdr && hdr->kind == VmObjKind::Array) {
                        VmArray* arr = static_cast<VmArray*>(hdr);
                        int64_t raw_idx = idx_v.is_int() ? idx_v.as_i64() : static_cast<int64_t>(idx_v.as_double());
                        if (raw_idx < 0) raw_idx += static_cast<int64_t>(arr->length);
                        size_t idx = static_cast<size_t>(raw_idx);
                        if (idx < arr->length) {
                            if (arr->elem_type == ValueType::F32) {
                                float* fptr = static_cast<float*>(arr->data);
                                float fv = static_cast<float>(rhs.to_number());
                                if (st->assign_op == "=") fptr[idx] = fv;
                                else if (st->assign_op == "+=") fptr[idx] += fv;
                                else if (st->assign_op == "*=") fptr[idx] *= fv;
                            } else if (arr->elem_type == ValueType::F64) {
                                double* dptr = static_cast<double*>(arr->data);
                                double dv = rhs.to_number();
                                if (st->assign_op == "=") dptr[idx] = dv;
                                else if (st->assign_op == "+=") dptr[idx] += dv;
                                else if (st->assign_op == "*=") dptr[idx] *= dv;
                            } else {
                                int64_t* iptr = static_cast<int64_t*>(arr->data);
                                int64_t iv = rhs.is_int() ? rhs.as_i64() : static_cast<int64_t>(rhs.as_double());
                                if (st->assign_op == "=") iptr[idx] = iv;
                                else if (st->assign_op == "+=") iptr[idx] += iv;
                                else if (st->assign_op == "*=") iptr[idx] *= iv;
                            }
                        }
                    }
                }
                break;
            }

            if (st->assign_op == "=") {
                env[st->name] = rhs;
            } else {
                rt::NaNVal cur = env.count(st->name) ? env[st->name] : globals_[st->name];
                if (cur.is_ptr() && rhs.is_ptr() && st->assign_op == "+=") {
                    env[st->name] = make_string(val_to_string(cur) + val_to_string(rhs));
                } else if (cur.is_int() && rhs.is_int()) {
                    int64_t a = cur.as_i64(), b = rhs.as_i64();
                    if (st->assign_op == "+=") env[st->name] = rt::NaNVal::from_i64(a + b);
                    else if (st->assign_op == "-=") env[st->name] = rt::NaNVal::from_i64(a - b);
                    else if (st->assign_op == "*=") env[st->name] = rt::NaNVal::from_i64(a * b);
                } else {
                    double a = cur.to_number(), b = rhs.to_number();
                    if (st->assign_op == "+=") env[st->name] = rt::NaNVal::from_double(a + b);
                    else if (st->assign_op == "-=") env[st->name] = rt::NaNVal::from_double(a - b);
                    else if (st->assign_op == "*=") env[st->name] = rt::NaNVal::from_double(a * b);
                    else if (st->assign_op == "/=") env[st->name] = rt::NaNVal::from_double(a / b);
                }
            }
            break;
        }
        case StmtKind::If: {
            rt::NaNVal c = eval_expr(st->expr.get(), env);
            bool cond = c.is_bool() ? (c.as_i64() != 0) : (c.to_number() != 0.0);
            if (cond) {
                for (const auto& b : st->body) {
                    exec_stmt(b.get(), env);
                    if (has_returned_ || has_break_ || has_continue_) break;
                }
            } else {
                for (const auto& eb : st->else_body) {
                    exec_stmt(eb.get(), env);
                    if (has_returned_ || has_break_ || has_continue_) break;
                }
            }
            break;
        }
        case StmtKind::ForRange: {
            int64_t start = eval_expr(st->range_start.get(), env).as_i64();
            int64_t stop  = eval_expr(st->range_stop.get(), env).as_i64();
            int64_t step  = eval_expr(st->range_step.get(), env).as_i64();

            for (int64_t i = start; i < stop; i += step) {
                env[st->loop_var] = rt::NaNVal::from_i64(i);
                for (const auto& b : st->body) {
                    exec_stmt(b.get(), env);
                    if (has_returned_ || has_break_ || has_continue_) break;
                }
                if (has_continue_) has_continue_ = false;
                if (has_break_) { has_break_ = false; break; }
                if (has_returned_) break;
            }
            break;
        }
        case StmtKind::ForEach: {
            rt::NaNVal col = eval_expr(st->expr.get(), env);
            if (col.is_ptr()) {
                VmHeader* hdr = static_cast<VmHeader*>(col.as_ptr());
                if (hdr && hdr->kind == VmObjKind::Array) {
                    VmArray* arr = static_cast<VmArray*>(hdr);
                    for (size_t i = 0; i < arr->length; ++i) {
                        if (arr->elem_type == ValueType::F32) env[st->loop_var] = rt::NaNVal::from_double(static_cast<float*>(arr->data)[i]);
                        else if (arr->elem_type == ValueType::F64) env[st->loop_var] = rt::NaNVal::from_double(static_cast<double*>(arr->data)[i]);
                        else env[st->loop_var] = rt::NaNVal::from_i64(static_cast<int64_t*>(arr->data)[i]);

                        for (const auto& b : st->body) {
                            exec_stmt(b.get(), env);
                            if (has_returned_ || has_break_ || has_continue_) break;
                        }
                        if (has_continue_) has_continue_ = false;
                        if (has_break_) { has_break_ = false; break; }
                        if (has_returned_) break;
                    }
                }
            }
            break;
        }
        case StmtKind::ForCStyle: {
            exec_stmt(st->c_init.get(), env);
            while (true) {
                rt::NaNVal c = eval_expr(st->c_cond.get(), env);
                bool cond = c.is_bool() ? (c.as_i64() != 0) : (c.to_number() != 0.0);
                if (!cond) break;
                for (const auto& b : st->body) {
                    exec_stmt(b.get(), env);
                    if (has_returned_ || has_break_ || has_continue_) break;
                }
                if (has_continue_) has_continue_ = false;
                if (has_break_) { has_break_ = false; break; }
                if (has_returned_) break;
                exec_stmt(st->c_step.get(), env);
            }
            break;
        }
        case StmtKind::While: {
            while (true) {
                rt::NaNVal c = eval_expr(st->expr.get(), env);
                bool cond = c.is_bool() ? (c.as_i64() != 0) : (c.to_number() != 0.0);
                if (!cond) break;
                for (const auto& b : st->body) {
                    exec_stmt(b.get(), env);
                    if (has_returned_ || has_break_ || has_continue_) break;
                }
                if (has_continue_) has_continue_ = false;
                if (has_break_) { has_break_ = false; break; }
                if (has_returned_) break;
            }
            break;
        }
        case StmtKind::Return: {
            return_val_ = st->expr ? eval_expr(st->expr.get(), env) : rt::NaNVal::null();
            has_returned_ = true;
            break;
        }
        case StmtKind::ExprStmt: {
            eval_expr(st->expr.get(), env);
            break;
        }
        default:
            break;
    }
}

rt::NaNVal BytecodeVM::eval_expr(const Expr* ex, std::unordered_map<std::string, rt::NaNVal>& env) {
    if (!ex) return rt::NaNVal::null();

    switch (ex->kind) {
        case ExprKind::IntLit:
            return rt::NaNVal::from_i64(ex->int_val);
        case ExprKind::FloatLit:
            return rt::NaNVal::from_double(ex->float_val);
        case ExprKind::BoolLit:
            return rt::NaNVal::from_bool(ex->bool_val);
        case ExprKind::NullLit:
            return rt::NaNVal::null();
        case ExprKind::StringLit:
            return make_string(ex->str_val);
        case ExprKind::FString: {
            std::string out;
            for (size_t i = 0; i < ex->keys.size(); ++i) {
                out += ex->keys[i];
                if (i < ex->args.size()) {
                    out += val_to_string(eval_expr(ex->args[i].get(), env));
                }
            }
            return make_string(out);
        }
        case ExprKind::Var: {
            auto it = env.find(ex->str_val);
            if (it != env.end()) return it->second;
            auto git = globals_.find(ex->str_val);
            if (git != globals_.end()) return git->second;
            return rt::NaNVal::from_i64(0);
        }
        case ExprKind::ArrayAlloc: {
            rt::NaNVal len_v = eval_expr(ex->lhs.get(), env);
            size_t n = static_cast<size_t>(len_v.is_int() ? len_v.as_i64() : static_cast<int64_t>(len_v.as_double()));
            VmArray* arr = static_cast<VmArray*>(arena_.alloc(sizeof(VmArray), 64));
            arr->kind = VmObjKind::Array;
            arr->elem_type = ex->alloc_elem_type;
            arr->length = n;
            arr->capacity = std::max<size_t>(n, 16);
            if (ex->alloc_elem_type == ValueType::F32) {
                float* buf = arena_.alloc_array<float>(arr->capacity);
                std::memset(buf, 0, n * sizeof(float));
                arr->data = buf;
            } else if (ex->alloc_elem_type == ValueType::F64) {
                double* buf = arena_.alloc_array<double>(arr->capacity);
                std::memset(buf, 0, n * sizeof(double));
                arr->data = buf;
            } else {
                int64_t* buf = arena_.alloc_array<int64_t>(arr->capacity);
                std::memset(buf, 0, n * sizeof(int64_t));
                arr->data = buf;
            }
            return rt::NaNVal::from_ptr(arr);
        }
        case ExprKind::ArrayLit: {
            size_t n = ex->args.size();
            size_t cap = std::max<size_t>(n * 2, 16);
            VmArray* arr = static_cast<VmArray*>(arena_.alloc(sizeof(VmArray), 64));
            arr->kind = VmObjKind::Array;
            arr->elem_type = ValueType::F64;
            arr->length = n;
            arr->capacity = cap;
            double* buf = arena_.alloc_array<double>(cap);
            for (size_t i = 0; i < n; ++i) {
                buf[i] = eval_expr(ex->args[i].get(), env).to_number();
            }
            arr->data = buf;
            return rt::NaNVal::from_ptr(arr);
        }
        case ExprKind::MapLit: {
            rt::NaNVal mv = make_map();
            VmMap* mp = static_cast<VmMap*>(mv.as_ptr());
            for (size_t i = 0; i < ex->keys.size() && i < ex->args.size(); ++i) {
                mp->set(ex->keys[i], eval_expr(ex->args[i].get(), env));
            }
            return mv;
        }
        case ExprKind::StructInit: {
            rt::NaNVal sv = make_map(ex->struct_type_name);
            VmMap* mp = static_cast<VmMap*>(sv.as_ptr());
            auto sit = structs_.find(ex->struct_type_name);
            if (sit != structs_.end()) {
                for (const auto& fld : sit->second->fields) {
                    if (fld.default_val) {
                        mp->set(fld.name, eval_expr(fld.default_val.get(), env));
                    } else {
                        mp->set(fld.name, rt::NaNVal::from_i64(0));
                    }
                }
            }
            for (size_t i = 0; i < ex->args.size(); ++i) {
                std::string k = (i < ex->keys.size() && !ex->keys[i].empty())
                    ? ex->keys[i]
                    : ((sit != structs_.end() && i < sit->second->fields.size()) ? sit->second->fields[i].name : std::to_string(i));
                mp->set(k, eval_expr(ex->args[i].get(), env));
            }
            return sv;
        }
        case ExprKind::FieldAccess: {
            if (ex->obj_name == "math") {
                if (ex->str_val == "pi" || ex->str_val == "PI") return rt::NaNVal::from_double(3.14159265358979323846);
                if (ex->str_val == "e" || ex->str_val == "E") return rt::NaNVal::from_double(2.71828182845904523536);
            }
            rt::NaNVal obj = eval_expr(ex->lhs.get(), env);
            if (obj.is_ptr()) {
                VmHeader* hdr = static_cast<VmHeader*>(obj.as_ptr());
                if (hdr && (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct)) {
                    return static_cast<VmMap*>(hdr)->get(ex->str_val);
                }
                if (hdr && hdr->kind == VmObjKind::Array && (ex->str_val == "len" || ex->str_val == "length")) {
                    return rt::NaNVal::from_i64(static_cast<int64_t>(static_cast<VmArray*>(hdr)->length));
                }
                if (hdr && hdr->kind == VmObjKind::String && (ex->str_val == "len" || ex->str_val == "length")) {
                    return rt::NaNVal::from_i64(static_cast<int64_t>(static_cast<VmString*>(hdr)->length));
                }
            }
            return rt::NaNVal::null();
        }
        case ExprKind::Index: {
            rt::NaNVal arr_v = eval_expr(ex->lhs.get(), env);
            rt::NaNVal idx_v = eval_expr(ex->rhs.get(), env);
            if (!arr_v.is_ptr()) return rt::NaNVal::from_i64(0);
            VmHeader* hdr = static_cast<VmHeader*>(arr_v.as_ptr());
            if (!hdr) return rt::NaNVal::from_i64(0);
            if (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct) {
                return static_cast<VmMap*>(hdr)->get(val_to_string(idx_v));
            }
            if (hdr->kind == VmObjKind::Array) {
                VmArray* arr = static_cast<VmArray*>(hdr);
                int64_t raw_idx = idx_v.is_int() ? idx_v.as_i64() : static_cast<int64_t>(idx_v.as_double());
                if (raw_idx < 0) raw_idx += static_cast<int64_t>(arr->length);
                size_t idx = static_cast<size_t>(raw_idx);
                if (idx >= arr->length) return rt::NaNVal::from_i64(0);
                if (arr->elem_type == ValueType::F32) return rt::NaNVal::from_double(static_cast<float*>(arr->data)[idx]);
                if (arr->elem_type == ValueType::F64) return rt::NaNVal::from_double(static_cast<double*>(arr->data)[idx]);
                return rt::NaNVal::from_i64(static_cast<int64_t*>(arr->data)[idx]);
            }
            return rt::NaNVal::from_i64(0);
        }
        case ExprKind::Ternary: {
            rt::NaNVal c = eval_expr(ex->cond_expr.get(), env);
            bool cond = c.is_bool() ? (c.as_i64() != 0) : (c.to_number() != 0.0);
            return cond ? eval_expr(ex->lhs.get(), env) : eval_expr(ex->rhs.get(), env);
        }
        case ExprKind::Unary: {
            rt::NaNVal v = eval_expr(ex->lhs.get(), env);
            if (ex->str_val == "-") {
                if (v.is_int()) return rt::NaNVal::from_i64(-v.as_i64());
                return rt::NaNVal::from_double(-v.as_double());
            }
            if (ex->str_val == "!") {
                bool b = v.is_bool() ? (v.as_i64() != 0) : (v.to_number() != 0.0);
                return rt::NaNVal::from_bool(!b);
            }
            return v;
        }
        case ExprKind::Binary: {
            const std::string& op = ex->str_val;
            if (op == "??") {
                rt::NaNVal l = eval_expr(ex->lhs.get(), env);
                if ((l.bits & rt::NaNVal::TAG_MASK) != rt::NaNVal::TAG_NULL) return l;
                return eval_expr(ex->rhs.get(), env);
            }
            rt::NaNVal l = eval_expr(ex->lhs.get(), env);
            rt::NaNVal r = eval_expr(ex->rhs.get(), env);

            if (l.is_ptr() || r.is_ptr()) {
                bool l_str = l.is_ptr() && l.as_ptr() && static_cast<VmHeader*>(l.as_ptr())->kind == VmObjKind::String;
                bool r_str = r.is_ptr() && r.as_ptr() && static_cast<VmHeader*>(r.as_ptr())->kind == VmObjKind::String;
                if (l_str || r_str) {
                    if (op == "+") return make_string(val_to_string(l) + val_to_string(r));
                    if (op == "==") return rt::NaNVal::from_bool(val_to_string(l) == val_to_string(r));
                    if (op == "!=") return rt::NaNVal::from_bool(val_to_string(l) != val_to_string(r));
                }
            }

            if (l.is_int() && r.is_int() && op != "/") {
                int64_t a = l.as_i64(), b = r.as_i64();
                if (op == "+") return rt::NaNVal::from_i64(a + b);
                if (op == "-") return rt::NaNVal::from_i64(a - b);
                if (op == "*") return rt::NaNVal::from_i64(a * b);
                if (op == "//") return rt::NaNVal::from_i64(b != 0 ? a / b : 0);
                if (op == "%") return rt::NaNVal::from_i64(b != 0 ? a % b : 0);
                if (op == "<<") return rt::NaNVal::from_i64(a << b);
                if (op == ">>") return rt::NaNVal::from_i64(a >> b);
                if (op == "&") return rt::NaNVal::from_i64(a & b);
                if (op == "|") return rt::NaNVal::from_i64(a | b);
                if (op == "^") return rt::NaNVal::from_i64(a ^ b);
                if (op == "<") return rt::NaNVal::from_bool(a < b);
                if (op == "<=") return rt::NaNVal::from_bool(a <= b);
                if (op == ">") return rt::NaNVal::from_bool(a > b);
                if (op == ">=") return rt::NaNVal::from_bool(a >= b);
                if (op == "==") return rt::NaNVal::from_bool(a == b);
                if (op == "!=") return rt::NaNVal::from_bool(a != b);
            }

            if (op == "&&") {
                bool lb = l.is_bool() ? (l.as_i64() != 0) : (l.to_number() != 0.0);
                bool rb = r.is_bool() ? (r.as_i64() != 0) : (r.to_number() != 0.0);
                return rt::NaNVal::from_bool(lb && rb);
            }
            if (op == "||") {
                bool lb = l.is_bool() ? (l.as_i64() != 0) : (l.to_number() != 0.0);
                bool rb = r.is_bool() ? (r.as_i64() != 0) : (r.to_number() != 0.0);
                return rt::NaNVal::from_bool(lb || rb);
            }

            double a = l.to_number(), b = r.to_number();
            if (op == "+") return rt::NaNVal::from_double(a + b);
            if (op == "-") return rt::NaNVal::from_double(a - b);
            if (op == "*") return rt::NaNVal::from_double(a * b);
            if (op == "/") return rt::NaNVal::from_double(a / b);
            if (op == "//") return rt::NaNVal::from_i64(static_cast<int64_t>(std::floor(a / b)));
            if (op == "**") return rt::NaNVal::from_double(std::pow(a, b));
            if (op == "<") return rt::NaNVal::from_bool(a < b);
            if (op == "<=") return rt::NaNVal::from_bool(a <= b);
            if (op == ">") return rt::NaNVal::from_bool(a > b);
            if (op == ">=") return rt::NaNVal::from_bool(a >= b);
            if (op == "==") return rt::NaNVal::from_bool(a == b);
            if (op == "!=") return rt::NaNVal::from_bool(a != b);
            return rt::NaNVal::from_double(0.0);
        }
        case ExprKind::Call: {
            if (ex->str_val == "print" || ex->str_val == "println") {
                for (size_t i = 0; i < ex->args.size(); ++i) {
                    if (i > 0) std::cout << " ";
                    print_val(eval_expr(ex->args[i].get(), env));
                }
                std::cout << "\n";
                return rt::NaNVal::null();
            }
            if (ex->str_val == "len" && !ex->args.empty()) {
                rt::NaNVal v = eval_expr(ex->args[0].get(), env);
                if (v.is_ptr() && v.as_ptr()) {
                    VmHeader* hdr = static_cast<VmHeader*>(v.as_ptr());
                    if (hdr->kind == VmObjKind::Array) return rt::NaNVal::from_i64(static_cast<VmArray*>(hdr)->length);
                    if (hdr->kind == VmObjKind::String) return rt::NaNVal::from_i64(static_cast<VmString*>(hdr)->length);
                    if (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct) return rt::NaNVal::from_i64(static_cast<VmMap*>(hdr)->entries.size());
                }
                return rt::NaNVal::from_i64(0);
            }
            if (ex->str_val == "str" && !ex->args.empty()) {
                return make_string(val_to_string(eval_expr(ex->args[0].get(), env)));
            }
            if (ex->str_val == "int" && !ex->args.empty()) {
                rt::NaNVal v = eval_expr(ex->args[0].get(), env);
                if (v.is_ptr()) return rt::NaNVal::from_i64(std::strtoll(val_to_string(v).c_str(), nullptr, 10));
                return rt::NaNVal::from_i64(static_cast<int64_t>(v.to_number()));
            }
            if (ex->str_val == "float" && !ex->args.empty()) {
                rt::NaNVal v = eval_expr(ex->args[0].get(), env);
                if (v.is_ptr()) return rt::NaNVal::from_double(std::strtod(val_to_string(v).c_str(), nullptr));
                return rt::NaNVal::from_double(v.to_number());
            }
            if (ex->str_val == "abs" && !ex->args.empty()) {
                rt::NaNVal v = eval_expr(ex->args[0].get(), env);
                if (v.is_int()) return rt::NaNVal::from_i64(std::abs(v.as_i64()));
                return rt::NaNVal::from_double(std::abs(v.to_number()));
            }
            if (ex->str_val == "min" && ex->args.size() >= 2) {
                double a = eval_expr(ex->args[0].get(), env).to_number();
                double b = eval_expr(ex->args[1].get(), env).to_number();
                return rt::NaNVal::from_double(std::min(a, b));
            }
            if (ex->str_val == "max" && ex->args.size() >= 2) {
                double a = eval_expr(ex->args[0].get(), env).to_number();
                double b = eval_expr(ex->args[1].get(), env).to_number();
                return rt::NaNVal::from_double(std::max(a, b));
            }
            if (ex->str_val == "clamp" && ex->args.size() >= 3) {
                double x = eval_expr(ex->args[0].get(), env).to_number();
                double lo = eval_expr(ex->args[1].get(), env).to_number();
                double hi = eval_expr(ex->args[2].get(), env).to_number();
                return rt::NaNVal::from_double(std::clamp(x, lo, hi));
            }
            if (ex->str_val == "type_of" && !ex->args.empty()) {
                rt::NaNVal v = eval_expr(ex->args[0].get(), env);
                if (v.is_int()) return make_string("i64");
                if (v.is_double()) return make_string("f64");
                if (v.is_bool()) return make_string("bool");
                if (v.is_ptr() && v.as_ptr()) {
                    VmHeader* hdr = static_cast<VmHeader*>(v.as_ptr());
                    if (hdr->kind == VmObjKind::String) return make_string("str");
                    if (hdr->kind == VmObjKind::Array) return make_string("array");
                    if (hdr->kind == VmObjKind::Map) return make_string("map");
                    if (hdr->kind == VmObjKind::Struct) return make_string(static_cast<VmMap*>(hdr)->type_name);
                }
                return make_string("null");
            }
            if (ex->str_val == "simd_dot" || ex->str_val == "parallel_dot") {
                VmArray* a = static_cast<VmArray*>(eval_expr(ex->args[0].get(), env).as_ptr());
                VmArray* b = static_cast<VmArray*>(eval_expr(ex->args[1].get(), env).as_ptr());
                size_t n = ex->args.size() > 2
                    ? static_cast<size_t>(eval_expr(ex->args[2].get(), env).as_i64())
                    : (a ? a->length : 0);
                if (a && b) {
                    if (a->elem_type == ValueType::F32) {
                        float res = (ex->str_val == "parallel_dot")
                            ? rt::parallel_simd_dot_f32(static_cast<const float*>(a->data), static_cast<const float*>(b->data), n)
                            : rt::simd_fma_dot_f32(static_cast<const float*>(a->data), static_cast<const float*>(b->data), n);
                        return rt::NaNVal::from_double(res);
                    } else if (a->elem_type == ValueType::F64) {
                        const double* da = static_cast<const double*>(a->data);
                        const double* db = static_cast<const double*>(b->data);
                        double s = 0.0;
                        for (size_t i = 0; i < n; ++i) s += da[i] * db[i];
                        return rt::NaNVal::from_double(s);
                    }
                }
            }
            if (ex->str_val == "fill_seq_f32" && ex->args.size() >= 3) {
                VmArray* a = static_cast<VmArray*>(eval_expr(ex->args[0].get(), env).as_ptr());
                float scale = static_cast<float>(eval_expr(ex->args[1].get(), env).to_number());
                size_t n = static_cast<size_t>(eval_expr(ex->args[2].get(), env).as_i64());
                if (a && a->elem_type == ValueType::F32) {
                    float* ptr = static_cast<float*>(a->data);
                    for (size_t i = 0; i < n; ++i) {
                        ptr[i] = static_cast<float>((i & 1023) + 1) * scale;
                    }
                }
                return rt::NaNVal::null();
            }
            if (ex->str_val == "init_matrix_ab_f32" && ex->args.size() >= 3) {
                VmArray* a = static_cast<VmArray*>(eval_expr(ex->args[0].get(), env).as_ptr());
                VmArray* b = static_cast<VmArray*>(eval_expr(ex->args[1].get(), env).as_ptr());
                size_t n = static_cast<size_t>(eval_expr(ex->args[2].get(), env).as_i64());
                if (a && b && a->elem_type == ValueType::F32 && b->elem_type == ValueType::F32) {
                    rt::init_matrix_ab_f32(static_cast<float*>(a->data), static_cast<float*>(b->data), n);
                }
                return rt::NaNVal::null();
            }
            if (ex->str_val == "matrix_fused_f32" && ex->args.size() >= 4) {
                VmArray* a = static_cast<VmArray*>(eval_expr(ex->args[0].get(), env).as_ptr());
                VmArray* b = static_cast<VmArray*>(eval_expr(ex->args[1].get(), env).as_ptr());
                VmArray* c = static_cast<VmArray*>(eval_expr(ex->args[2].get(), env).as_ptr());
                size_t n = static_cast<size_t>(eval_expr(ex->args[3].get(), env).as_i64());
                if (a && b && c) {
                    double res = rt::matrix_fused_process_f32(
                        static_cast<const float*>(a->data),
                        static_cast<const float*>(b->data),
                        static_cast<float*>(c->data),
                        n);
                    return rt::NaNVal::from_double(res);
                }
                return rt::NaNVal::from_double(0.0);
            }
            if (ex->str_val == "matrix_matmul_f32" && ex->args.size() >= 4) {
                VmArray* a = static_cast<VmArray*>(eval_expr(ex->args[0].get(), env).as_ptr());
                VmArray* b = static_cast<VmArray*>(eval_expr(ex->args[1].get(), env).as_ptr());
                VmArray* c = static_cast<VmArray*>(eval_expr(ex->args[2].get(), env).as_ptr());
                size_t n = static_cast<size_t>(eval_expr(ex->args[3].get(), env).as_i64());
                if (a && b && c) {
                    double res = rt::matrix_matmul_avx2_f32(
                        static_cast<const float*>(a->data),
                        static_cast<const float*>(b->data),
                        static_cast<float*>(c->data),
                        n);
                    return rt::NaNVal::from_double(res);
                }
                return rt::NaNVal::from_double(0.0);
            }
            if (ex->str_val == "py_exec" && ex->args.size() >= 1) {
                std::string code = val_to_string(eval_expr(ex->args[0].get(), env));
                py::bridge().exec_python_block(code);
                return rt::NaNVal::null();
            }
            if (ex->str_val == "py_eval_f64" && ex->args.size() >= 1) {
                std::string code = val_to_string(eval_expr(ex->args[0].get(), env));
                return rt::NaNVal::from_double(py::bridge().eval_to_double(code));
            }
            if (ex->str_val == "clock_ms") {
                auto now = std::chrono::steady_clock::now().time_since_epoch();
                double ms = std::chrono::duration<double, std::milli>(now).count();
                return rt::NaNVal::from_double(ms);
            }
            if (ex->str_val == "arena_kb") {
                return rt::NaNVal::from_double(static_cast<double>(arena_.total_allocated_bytes()) / 1024.0);
            }

            // Execute compiled function on the Register Bytecode VM!
            auto cfit = compiled_funcs_.find(ex->str_val);
            if (cfit != compiled_funcs_.end()) {
                std::vector<rt::NaNVal> arg_vals(ex->args.size());
                for (size_t i = 0; i < ex->args.size(); ++i) {
                    arg_vals[i] = eval_expr(ex->args[i].get(), env);
                }
                return run_compiled_function(cfit->second, arg_vals.data(), arg_vals.size());
            }

            auto fit = functions_.find(ex->str_val);
            if (fit != functions_.end()) {
                compiled_funcs_[ex->str_val] = compile_function(fit->second);
                std::vector<rt::NaNVal> arg_vals(ex->args.size());
                for (size_t i = 0; i < ex->args.size(); ++i) {
                    arg_vals[i] = eval_expr(ex->args[i].get(), env);
                }
                return run_compiled_function(compiled_funcs_[ex->str_val], arg_vals.data(), arg_vals.size());
            }
            return rt::NaNVal::null();
        }
        case ExprKind::MemberCall: {
            std::vector<rt::NaNVal> arg_vals;
            for (const auto& a : ex->args) {
                arg_vals.push_back(eval_expr(a.get(), env));
            }

            if (ex->obj_name == "math") {
                if (arg_vals.size() == 1) {
                    double x = arg_vals[0].to_number();
                    if (ex->str_val == "sqrt") return rt::NaNVal::from_double(std::sqrt(x));
                    if (ex->str_val == "sin") return rt::NaNVal::from_double(std::sin(x));
                    if (ex->str_val == "cos") return rt::NaNVal::from_double(std::cos(x));
                    if (ex->str_val == "tan") return rt::NaNVal::from_double(std::tan(x));
                    if (ex->str_val == "exp") return rt::NaNVal::from_double(std::exp(x));
                    if (ex->str_val == "log") return rt::NaNVal::from_double(std::log(x));
                    if (ex->str_val == "floor") return rt::NaNVal::from_i64(static_cast<int64_t>(std::floor(x)));
                    if (ex->str_val == "ceil") return rt::NaNVal::from_i64(static_cast<int64_t>(std::ceil(x)));
                    if (ex->str_val == "round") return rt::NaNVal::from_i64(static_cast<int64_t>(std::llround(x)));
                }
                if (arg_vals.size() == 2) {
                    double a = arg_vals[0].to_number(), b = arg_vals[1].to_number();
                    if (ex->str_val == "hypot") return rt::NaNVal::from_double(std::hypot(a, b));
                    if (ex->str_val == "pow") return rt::NaNVal::from_double(std::pow(a, b));
                }
            }
            if (ex->obj_name == "fs") {
                if (ex->str_val == "read" && !arg_vals.empty()) {
                    std::ifstream ifs(val_to_string(arg_vals[0]), std::ios::binary);
                    std::ostringstream ss;
                    ss << ifs.rdbuf();
                    return make_string(ss.str());
                }
                if (ex->str_val == "write" && arg_vals.size() >= 2) {
                    std::ofstream ofs(val_to_string(arg_vals[0]), std::ios::binary);
                    ofs << val_to_string(arg_vals[1]);
                    return rt::NaNVal::from_bool(static_cast<bool>(ofs));
                }
                if (ex->str_val == "exists" && !arg_vals.empty()) {
                    return rt::NaNVal::from_bool(fs::exists(val_to_string(arg_vals[0])));
                }
            }
            if (ex->obj_name == "json") {
                if ((ex->str_val == "stringify" || ex->str_val == "dumps") && !arg_vals.empty()) {
                    return make_string(val_to_string(arg_vals[0]));
                }
            }
            if (ex->obj_name == "env") {
                if (ex->str_val == "get" && !arg_vals.empty()) {
                    const char* v = std::getenv(val_to_string(arg_vals[0]).c_str());
                    if (v) return make_string(v);
                    if (arg_vals.size() >= 2) return arg_vals[1];
                    return make_string("");
                }
            }
            if (ex->obj_name == "crypto") {
                if (ex->str_val == "fnv1a64" && !arg_vals.empty()) {
                    return rt::NaNVal::from_i64(rt::fnv1a64_hash(val_to_string(arg_vals[0])));
                }
            }

            if (ex->lhs) {
                rt::NaNVal target = eval_expr(ex->lhs.get(), env);
                if (target.is_ptr() && target.as_ptr()) {
                    VmHeader* hdr = static_cast<VmHeader*>(target.as_ptr());

                    if (hdr->kind == VmObjKind::Array) {
                        VmArray* arr = static_cast<VmArray*>(hdr);
                        if (ex->str_val == "len" || ex->str_val == "size") {
                            return rt::NaNVal::from_i64(static_cast<int64_t>(arr->length));
                        }
                        if (ex->str_val == "sum") {
                            double s = 0.0;
                            if (arr->elem_type == ValueType::F32) {
                                const float* p = static_cast<const float*>(arr->data);
                                for (size_t i = 0; i < arr->length; ++i) s += p[i];
                            } else if (arr->elem_type == ValueType::F64) {
                                const double* p = static_cast<const double*>(arr->data);
                                for (size_t i = 0; i < arr->length; ++i) s += p[i];
                            } else {
                                return rt::NaNVal::from_i64(rt::simd_sum_i64(static_cast<const int64_t*>(arr->data), arr->length));
                            }
                            return rt::NaNVal::from_double(s);
                        }
                        if (ex->str_val == "mean" || ex->str_val == "avg") {
                            if (arr->length == 0) return rt::NaNVal::from_double(0.0);
                            double s = 0.0;
                            if (arr->elem_type == ValueType::F32) {
                                const float* p = static_cast<const float*>(arr->data);
                                for (size_t i = 0; i < arr->length; ++i) s += p[i];
                            } else if (arr->elem_type == ValueType::F64) {
                                const double* p = static_cast<const double*>(arr->data);
                                for (size_t i = 0; i < arr->length; ++i) s += p[i];
                            } else {
                                s = static_cast<double>(rt::simd_sum_i64(static_cast<const int64_t*>(arr->data), arr->length));
                            }
                            return rt::NaNVal::from_double(s / static_cast<double>(arr->length));
                        }
                        if (ex->str_val == "min" || ex->str_val == "max") {
                            if (arr->length == 0) return rt::NaNVal::from_double(0.0);
                            bool is_min = (ex->str_val == "min");
                            double best = (arr->elem_type == ValueType::F32) ? static_cast<const float*>(arr->data)[0]
                                        : (arr->elem_type == ValueType::F64) ? static_cast<const double*>(arr->data)[0]
                                        : static_cast<double>(static_cast<const int64_t*>(arr->data)[0]);
                            for (size_t i = 1; i < arr->length; ++i) {
                                double v = (arr->elem_type == ValueType::F32) ? static_cast<const float*>(arr->data)[i]
                                         : (arr->elem_type == ValueType::F64) ? static_cast<const double*>(arr->data)[i]
                                         : static_cast<double>(static_cast<const int64_t*>(arr->data)[i]);
                                if (is_min && v < best) best = v;
                                if (!is_min && v > best) best = v;
                            }
                            return rt::NaNVal::from_double(best);
                        }
                        if (ex->str_val == "dot" && !arg_vals.empty() && arg_vals[0].is_ptr()) {
                            VmArray* other = static_cast<VmArray*>(arg_vals[0].as_ptr());
                            size_t n = std::min(arr->length, other->length);
                            if (arr->elem_type == ValueType::F32 && other->elem_type == ValueType::F32) {
                                return rt::NaNVal::from_double(rt::parallel_simd_dot_f32(
                                    static_cast<const float*>(arr->data),
                                    static_cast<const float*>(other->data), n));
                            } else if (arr->elem_type == ValueType::F64 && other->elem_type == ValueType::F64) {
                                const double* da = static_cast<const double*>(arr->data);
                                const double* db = static_cast<const double*>(other->data);
                                double s = 0.0;
                                for (size_t i = 0; i < n; ++i) s += da[i] * db[i];
                                return rt::NaNVal::from_double(s);
                            }
                        }
                        if ((ex->str_val == "push" || ex->str_val == "append") && !arg_vals.empty()) {
                            if (arr->length >= arr->capacity) {
                                size_t new_cap = std::max<size_t>(arr->capacity * 2, 16);
                                if (arr->elem_type == ValueType::F64) {
                                    double* nb = arena_.alloc_array<double>(new_cap);
                                    std::memcpy(nb, arr->data, arr->length * sizeof(double));
                                    arr->data = nb;
                                } else if (arr->elem_type == ValueType::F32) {
                                    float* nb = arena_.alloc_array<float>(new_cap);
                                    std::memcpy(nb, arr->data, arr->length * sizeof(float));
                                    arr->data = nb;
                                } else {
                                    int64_t* nb = arena_.alloc_array<int64_t>(new_cap);
                                    std::memcpy(nb, arr->data, arr->length * sizeof(int64_t));
                                    arr->data = nb;
                                }
                                arr->capacity = new_cap;
                            }
                            if (arr->elem_type == ValueType::F64) static_cast<double*>(arr->data)[arr->length++] = arg_vals[0].to_number();
                            else if (arr->elem_type == ValueType::F32) static_cast<float*>(arr->data)[arr->length++] = static_cast<float>(arg_vals[0].to_number());
                            else static_cast<int64_t*>(arr->data)[arr->length++] = arg_vals[0].as_i64();
                            return target;
                        }
                    }

                    if (hdr->kind == VmObjKind::String) {
                        std::string s = val_to_string(target);
                        if (ex->str_val == "len" || ex->str_val == "size") return rt::NaNVal::from_i64(static_cast<int64_t>(s.size()));
                        if (ex->str_val == "upper") return make_string(rt::str_upper(s));
                        if (ex->str_val == "lower") return make_string(rt::str_lower(s));
                        if (ex->str_val == "trim" || ex->str_val == "strip") return make_string(rt::str_trim(s));
                        if (ex->str_val == "contains" && !arg_vals.empty()) return rt::NaNVal::from_bool(rt::str_contains(s, val_to_string(arg_vals[0])));
                        if (ex->str_val == "starts_with" && !arg_vals.empty()) return rt::NaNVal::from_bool(rt::str_starts_with(s, val_to_string(arg_vals[0])));
                        if (ex->str_val == "replace" && arg_vals.size() >= 2) return make_string(rt::str_replace(s, val_to_string(arg_vals[0]), val_to_string(arg_vals[1])));
                    }

                    if (hdr->kind == VmObjKind::Map || hdr->kind == VmObjKind::Struct) {
                        VmMap* mp = static_cast<VmMap*>(hdr);
                        if ((ex->str_val == "has" || ex->str_val == "contains") && !arg_vals.empty()) {
                            return rt::NaNVal::from_bool(mp->contains(val_to_string(arg_vals[0])));
                        }
                        if (ex->str_val == "get" && !arg_vals.empty()) {
                            std::string k = val_to_string(arg_vals[0]);
                            if (mp->contains(k)) return mp->get(k);
                            if (arg_vals.size() >= 2) return arg_vals[1];
                            return rt::NaNVal::null();
                        }
                    }
                }
            }

            return py::bridge().call_module_fn(ex->obj_name, ex->str_val, arg_vals);
        }
    }
    return rt::NaNVal::null();
}

void BytecodeVM::print_val(rt::NaNVal v) {
    std::cout << val_to_string(v);
}

} // namespace flawn
