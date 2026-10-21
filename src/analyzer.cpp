#include "flawn_analyzer.hpp"

namespace flawn {

AnalysisStats Analyzer::analyze(Program& prog) {
    stats_ = AnalysisStats{};
    sym_table_.clear();
    for (auto& st : prog.statements) {
        analyze_stmt(st.get());
    }
    return stats_;
}

void Analyzer::analyze_stmt(Stmt* st) {
    if (!st) return;

    switch (st->kind) {
        case StmtKind::StructDecl: {
            sym_table_[st->name] = ValueType::Struct;
            break;
        }
        case StmtKind::VarDecl: {
            ValueType rhs_t = analyze_expr(st->expr.get());
            if (st->decl_type == ValueType::Inferred) {
                st->decl_type = (rhs_t != ValueType::Inferred) ? rhs_t : ValueType::I64;
            }
            if (st->expr && st->expr->kind == ExprKind::StructInit && st->struct_type_name.empty()) {
                st->struct_type_name = st->expr->struct_type_name;
            }
            if (st->decl_type == ValueType::SliceF32 || st->decl_type == ValueType::SliceF64 ||
                st->decl_type == ValueType::SliceI64) {
                stats_.arena_buffers_allocated++;
            }
            sym_table_[st->name] = st->decl_type;
            break;
        }
        case StmtKind::Assign: {
            ValueType rhs_t = analyze_expr(st->expr.get());
            if (st->index_expr) {
                analyze_expr(st->index_expr.get());
            } else if (!st->field_name.empty()) {
                // struct field assignment `obj.field = val`
            } else if (st->assign_op == "=" && sym_table_.find(st->name) == sym_table_.end()) {
                // Python implicit variable declaration (`total = 0` -> lift to VarDecl)
                st->kind = StmtKind::VarDecl;
                st->decl_type = (rhs_t != ValueType::Inferred) ? rhs_t : ValueType::I64;
                if (st->expr && st->expr->kind == ExprKind::StructInit) {
                    st->struct_type_name = st->expr->struct_type_name;
                }
                if (st->decl_type == ValueType::SliceF32 || st->decl_type == ValueType::SliceF64 ||
                    st->decl_type == ValueType::SliceI64) {
                    stats_.arena_buffers_allocated++;
                }
                sym_table_[st->name] = st->decl_type;
            }
            break;
        }
        case StmtKind::FuncDecl: {
            auto saved_syms = sym_table_;
            if (st->is_python_def) {
                stats_.python_funcs_unboxed++;
            } else {
                stats_.flawn_funcs_compiled++;
            }
            for (auto& p : st->params) {
                if (p.default_val) analyze_expr(p.default_val.get());
                if (p.type == ValueType::Inferred) {
                    p.type = p.default_val ? p.default_val->inferred_type : ValueType::I64;
                }
                sym_table_[p.name] = p.type;
            }
            for (auto& body_st : st->body) {
                if (st->attr_simd && (body_st->kind == StmtKind::ForCStyle || body_st->kind == StmtKind::ForRange)) {
                    body_st->attr_simd = true;
                    body_st->attr_parallel |= st->attr_parallel;
                }
                analyze_stmt(body_st.get());
            }
            if (st->return_type == ValueType::Inferred) {
                st->return_type = ValueType::Void;
                for (auto& body_st : st->body) {
                    if (body_st && body_st->kind == StmtKind::Return && body_st->expr) {
                        ValueType rt = body_st->expr->inferred_type;
                        st->return_type = (rt != ValueType::Inferred) ? rt : ValueType::I64;
                    }
                }
            }
            sym_table_ = std::move(saved_syms);
            sym_table_[st->name] = st->return_type;
            break;
        }
        case StmtKind::ForRange: {
            analyze_expr(st->range_start.get());
            analyze_expr(st->range_stop.get());
            analyze_expr(st->range_step.get());
            sym_table_[st->loop_var] = ValueType::I64;
            for (auto& b : st->body) analyze_stmt(b.get());
            inspect_loop_optimizations(st);
            break;
        }
        case StmtKind::ForEach: {
            ValueType col_t = analyze_expr(st->expr.get());
            if (col_t == ValueType::SliceF32) sym_table_[st->loop_var] = ValueType::F32;
            else if (col_t == ValueType::SliceI64) sym_table_[st->loop_var] = ValueType::I64;
            else sym_table_[st->loop_var] = ValueType::F64;
            for (auto& b : st->body) analyze_stmt(b.get());
            break;
        }
        case StmtKind::ForCStyle: {
            analyze_stmt(st->c_init.get());
            analyze_expr(st->c_cond.get());
            analyze_stmt(st->c_step.get());
            for (auto& b : st->body) analyze_stmt(b.get());
            inspect_loop_optimizations(st);
            break;
        }
        case StmtKind::If: {
            analyze_expr(st->expr.get());
            for (auto& b : st->body) analyze_stmt(b.get());
            for (auto& eb : st->else_body) analyze_stmt(eb.get());
            break;
        }
        case StmtKind::While: {
            analyze_expr(st->expr.get());
            for (auto& b : st->body) analyze_stmt(b.get());
            break;
        }
        case StmtKind::Assert: {
            analyze_expr(st->expr.get());
            if (st->assert_msg) analyze_expr(st->assert_msg.get());
            break;
        }
        case StmtKind::Return:
        case StmtKind::ExprStmt: {
            analyze_expr(st->expr.get());
            break;
        }
        default:
            break;
    }
}

void Analyzer::inspect_loop_optimizations(Stmt* loop_stmt) {
    if (loop_stmt->attr_simd) {
        stats_.simd_loops_vectorized++;
    }
    if (loop_stmt->attr_parallel) {
        stats_.parallel_kernels++;
    }

    // Only lift to `simd_fma_dot_f32` when loop is explicitly annotated `@simd` and indices match the loop induction variable
    std::string ivar = loop_stmt->loop_var;
    if (ivar.empty() && loop_stmt->c_init) ivar = loop_stmt->c_init->name;

    if (loop_stmt->attr_simd && loop_stmt->body.size() == 1 && !ivar.empty()) {
        Stmt* body0 = loop_stmt->body[0].get();
        if (body0 && body0->kind == StmtKind::Assign && body0->assign_op == "+=" &&
            body0->expr && body0->expr->kind == ExprKind::Binary && body0->expr->str_val == "*") {
            Expr* lhs = body0->expr->lhs.get();
            Expr* rhs = body0->expr->rhs.get();
            if (lhs && rhs && lhs->kind == ExprKind::Index && rhs->kind == ExprKind::Index &&
                lhs->lhs && rhs->lhs && lhs->lhs->kind == ExprKind::Var && rhs->lhs->kind == ExprKind::Var &&
                lhs->rhs && rhs->rhs && lhs->rhs->kind == ExprKind::Var && rhs->rhs->kind == ExprKind::Var &&
                lhs->rhs->str_val == ivar && rhs->rhs->str_val == ivar) {
                loop_stmt->can_lift_simd_dot = true;
                loop_stmt->dot_arr_a = lhs->lhs->str_val;
                loop_stmt->dot_arr_b = rhs->lhs->str_val;
            }
        }
    }
}

ValueType Analyzer::analyze_expr(Expr* ex) {
    if (!ex) return ValueType::Void;

    switch (ex->kind) {
        case ExprKind::IntLit:
            ex->inferred_type = ValueType::I64;
            return ValueType::I64;
        case ExprKind::FloatLit:
            ex->inferred_type = ValueType::F64;
            return ValueType::F64;
        case ExprKind::BoolLit:
            ex->inferred_type = ValueType::Bool;
            return ValueType::Bool;
        case ExprKind::NullLit:
            ex->inferred_type = ValueType::Void;
            return ValueType::Void;
        case ExprKind::StringLit:
            ex->inferred_type = ValueType::String;
            return ValueType::String;
        case ExprKind::FString:
            for (auto& a : ex->args) analyze_expr(a.get());
            ex->inferred_type = ValueType::String;
            return ValueType::String;
        case ExprKind::Var: {
            auto it = sym_table_.find(ex->str_val);
            if (it != sym_table_.end()) {
                ex->inferred_type = it->second;
                return it->second;
            }
            ex->inferred_type = ValueType::I64;
            return ValueType::I64;
        }
        case ExprKind::ArrayAlloc: {
            analyze_expr(ex->lhs.get());
            if (ex->alloc_elem_type == ValueType::F32) ex->inferred_type = ValueType::SliceF32;
            else if (ex->alloc_elem_type == ValueType::F64) ex->inferred_type = ValueType::SliceF64;
            else ex->inferred_type = ValueType::SliceI64;
            return ex->inferred_type;
        }
        case ExprKind::ArrayLit: {
            for (auto& a : ex->args) analyze_expr(a.get());
            ex->inferred_type = ValueType::SliceF64;
            return ex->inferred_type;
        }
        case ExprKind::MapLit: {
            for (auto& a : ex->args) analyze_expr(a.get());
            ex->inferred_type = ValueType::Map;
            return ValueType::Map;
        }
        case ExprKind::StructInit: {
            for (auto& a : ex->args) analyze_expr(a.get());
            ex->inferred_type = ValueType::Struct;
            return ValueType::Struct;
        }
        case ExprKind::FieldAccess: {
            analyze_expr(ex->lhs.get());
            ex->inferred_type = ValueType::F64;
            return ValueType::F64;
        }
        case ExprKind::Index: {
            ValueType arr_t = analyze_expr(ex->lhs.get());
            analyze_expr(ex->rhs.get());
            if (arr_t == ValueType::SliceF32) ex->inferred_type = ValueType::F32;
            else if (arr_t == ValueType::SliceF64) ex->inferred_type = ValueType::F64;
            else ex->inferred_type = ValueType::I64;
            return ex->inferred_type;
        }
        case ExprKind::Unary: {
            ex->inferred_type = analyze_expr(ex->lhs.get());
            return ex->inferred_type;
        }
        case ExprKind::Ternary: {
            analyze_expr(ex->cond_expr.get());
            ValueType lt = analyze_expr(ex->lhs.get());
            analyze_expr(ex->rhs.get());
            ex->inferred_type = lt;
            return lt;
        }
        case ExprKind::Binary: {
            ValueType lt = analyze_expr(ex->lhs.get());
            ValueType rt = analyze_expr(ex->rhs.get());
            if (ex->str_val == "+" && (lt == ValueType::String || rt == ValueType::String)) {
                ex->inferred_type = ValueType::String;
                return ValueType::String;
            }
            if (ex->str_val == "==" || ex->str_val == "!=" || ex->str_val == "<" ||
                ex->str_val == "<=" || ex->str_val == ">" || ex->str_val == ">=" ||
                ex->str_val == "&&" || ex->str_val == "||") {
                ex->inferred_type = ValueType::Bool;
                return ValueType::Bool;
            }
            if (lt == ValueType::F64 || rt == ValueType::F64 || ex->str_val == "/") {
                if (ex->str_val == "/" && lt == ValueType::F32 && rt == ValueType::F32) {
                    ex->inferred_type = ValueType::F32;
                    return ValueType::F32;
                }
                ex->inferred_type = ValueType::F64;
                return ValueType::F64;
            }
            if (lt == ValueType::F32 || rt == ValueType::F32) {
                ex->inferred_type = ValueType::F32;
                return ValueType::F32;
            }
            ex->inferred_type = ValueType::I64;
            return ValueType::I64;
        }
        case ExprKind::Call: {
            for (auto& a : ex->args) analyze_expr(a.get());
            if (ex->str_val == "simd_dot" || ex->str_val == "parallel_dot") {
                ex->inferred_type = ValueType::F32;
                return ValueType::F32;
            }
            if (ex->str_val == "str" || ex->str_val == "read_file" || ex->str_val == "json_stringify" ||
                ex->str_val == "env_get" || ex->str_val == "type_of") {
                ex->inferred_type = ValueType::String;
                return ValueType::String;
            }
            if (ex->str_val == "len" || ex->str_val == "int") {
                ex->inferred_type = ValueType::I64;
                return ValueType::I64;
            }
            auto it = sym_table_.find(ex->str_val);
            if (it != sym_table_.end()) {
                ex->inferred_type = it->second;
                return it->second;
            }
            ex->inferred_type = ValueType::F64;
            return ValueType::F64;
        }
        case ExprKind::MemberCall: {
            if (ex->lhs) analyze_expr(ex->lhs.get());
            for (auto& a : ex->args) analyze_expr(a.get());
            if (ex->str_val == "upper" || ex->str_val == "lower" || ex->str_val == "trim" ||
                ex->str_val == "replace" || ex->str_val == "join" || ex->str_val == "read" ||
                ex->str_val == "stringify") {
                ex->inferred_type = ValueType::String;
                return ValueType::String;
            }
            if (ex->str_val == "len" || ex->str_val == "size" || ex->str_val == "count") {
                ex->inferred_type = ValueType::I64;
                return ValueType::I64;
            }
            if (ex->str_val == "contains" || ex->str_val == "starts_with" || ex->str_val == "ends_with" ||
                ex->str_val == "exists") {
                ex->inferred_type = ValueType::Bool;
                return ValueType::Bool;
            }
            ex->inferred_type = ValueType::F64;
            return ValueType::F64;
        }
    }
    return ValueType::I64;
}

} // namespace flawn
