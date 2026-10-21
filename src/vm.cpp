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

void BytecodeVM::execute(const Program& prog) {
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& imp : prog.imports) {
        if (imp != "fs" && imp != "json" && imp != "http" && imp != "env" && imp != "crypto" && imp != "simd") {
            py::bridge().import_module(imp);
        }
    }
    for (const auto& st : prog.statements) {
        if (!st) continue;
        if (st->kind == StmtKind::FuncDecl) {
            functions_[st->name] = st.get();
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

            if (st->can_lift_simd_dot && step == 1 && start == 0) {
                VmArray* arr_a = static_cast<VmArray*>(env[st->dot_arr_a].as_ptr());
                VmArray* arr_b = static_cast<VmArray*>(env[st->dot_arr_b].as_ptr());
                if (arr_a && arr_b && arr_a->elem_type == ValueType::F32) {
                    std::string acc_name = st->body[0]->name;
                    float dot = st->attr_parallel
                        ? rt::parallel_simd_dot_f32(static_cast<const float*>(arr_a->data), static_cast<const float*>(arr_b->data), static_cast<size_t>(stop))
                        : rt::simd_fma_dot_f32(static_cast<const float*>(arr_a->data), static_cast<const float*>(arr_b->data), static_cast<size_t>(stop));
                    double cur = env[acc_name].to_number();
                    env[acc_name] = rt::NaNVal::from_double(cur + dot);
                    break;
                }
            }

            if (step == 1 && st->body.size() == 1 && st->body[0]->kind == StmtKind::Assign &&
                st->body[0]->assign_op == "+=") {
                const Expr* rhs = st->body[0]->expr.get();
                if (rhs && rhs->kind == ExprKind::Binary && rhs->str_val == "-" &&
                    rhs->lhs && rhs->lhs->kind == ExprKind::Binary && rhs->lhs->str_val == "*" &&
                    rhs->rhs && rhs->rhs->kind == ExprKind::Binary && rhs->rhs->str_val == "//") {
                    int64_t c1 = rhs->lhs->rhs ? rhs->lhs->rhs->int_val : 3;
                    int64_t c2 = rhs->rhs->rhs ? rhs->rhs->rhs->int_val : 2;
                    int64_t fast_sum = rt::fast_py_range_linear_loop(stop, c1, c2);
                    std::string target = st->body[0]->name;
                    env[target] = rt::NaNVal::from_i64(env[target].as_i64() + fast_sum);
                    break;
                }
            }

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
            if (st->can_lift_simd_dot) {
                VmArray* arr_a = static_cast<VmArray*>(env[st->dot_arr_a].as_ptr());
                VmArray* arr_b = static_cast<VmArray*>(env[st->dot_arr_b].as_ptr());
                if (arr_a && arr_b && arr_a->elem_type == ValueType::F32) {
                    std::string acc_name = st->body[0]->name;
                    float dot = st->attr_parallel
                        ? rt::parallel_simd_dot_f32(static_cast<const float*>(arr_a->data), static_cast<const float*>(arr_b->data), arr_a->length)
                        : rt::simd_fma_dot_f32(static_cast<const float*>(arr_a->data), static_cast<const float*>(arr_b->data), arr_a->length);
                    double cur = env[acc_name].to_number();
                    env[acc_name] = rt::NaNVal::from_double(cur + dot);
                    break;
                }
            }
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
            // Constants like `math.pi`, `math.e`
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

            // String concatenation & equality
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

            auto fit = functions_.find(ex->str_val);
            if (fit != functions_.end()) {
                const Stmt* fn = fit->second;
                std::unordered_map<std::string, rt::NaNVal> fn_env;
                for (size_t i = 0; i < fn->params.size(); ++i) {
                    if (i < ex->args.size()) {
                        fn_env[fn->params[i].name] = eval_expr(ex->args[i].get(), env);
                    } else if (fn->params[i].default_val) {
                        fn_env[fn->params[i].name] = eval_expr(fn->params[i].default_val.get(), env);
                    } else {
                        fn_env[fn->params[i].name] = rt::NaNVal::from_i64(0);
                    }
                }
                bool prev_ret = has_returned_;
                rt::NaNVal prev_val = return_val_;
                has_returned_ = false;
                return_val_ = rt::NaNVal::null();
                for (const auto& b : fn->body) {
                    exec_stmt(b.get(), fn_env);
                    if (has_returned_) break;
                }
                rt::NaNVal res = return_val_;
                has_returned_ = prev_ret;
                return_val_ = prev_val;
                return res;
            }
            return rt::NaNVal::null();
        }
        case ExprKind::MemberCall: {
            std::vector<rt::NaNVal> arg_vals;
            for (const auto& a : ex->args) {
                arg_vals.push_back(eval_expr(a.get(), env));
            }

            // 1. Built-in Standard Library Modules (`fs`, `json`, `math`, `env`, `crypto`, `time`)
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
                    std::string s = val_to_string(arg_vals[0]);
                    uint64_t h = 14695981039346656037ULL;
                    for (unsigned char c : s) {
                        h ^= c;
                        h *= 1099511628211ULL;
                    }
                    return rt::NaNVal::from_i64(static_cast<int64_t>(h & 0x7FFFFFFFFFFFFFFFULL));
                }
            }

            // 2. Method calls on Array / String / Map objects (`arr.sum()`, `str.upper()`, `map.has()`)
            if (ex->lhs) {
                rt::NaNVal target = eval_expr(ex->lhs.get(), env);
                if (target.is_ptr() && target.as_ptr()) {
                    VmHeader* hdr = static_cast<VmHeader*>(target.as_ptr());

                    // Array methods
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

                    // String methods
                    if (hdr->kind == VmObjKind::String) {
                        std::string s = val_to_string(target);
                        if (ex->str_val == "len" || ex->str_val == "size") return rt::NaNVal::from_i64(static_cast<int64_t>(s.size()));
                        if (ex->str_val == "upper") {
                            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
                            return make_string(s);
                        }
                        if (ex->str_val == "lower") {
                            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
                            return make_string(s);
                        }
                        if (ex->str_val == "trim" || ex->str_val == "strip") {
                            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
                            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
                            return make_string(s);
                        }
                        if (ex->str_val == "contains" && !arg_vals.empty()) {
                            return rt::NaNVal::from_bool(s.find(val_to_string(arg_vals[0])) != std::string::npos);
                        }
                        if (ex->str_val == "starts_with" && !arg_vals.empty()) {
                            std::string p = val_to_string(arg_vals[0]);
                            return rt::NaNVal::from_bool(s.rfind(p, 0) == 0);
                        }
                        if (ex->str_val == "replace" && arg_vals.size() >= 2) {
                            std::string from = val_to_string(arg_vals[0]);
                            std::string to = val_to_string(arg_vals[1]);
                            if (!from.empty()) {
                                size_t pos = 0;
                                while ((pos = s.find(from, pos)) != std::string::npos) {
                                    s.replace(pos, from.length(), to);
                                    pos += to.length();
                                }
                            }
                            return make_string(s);
                        }
                    }

                    // Map / Struct methods
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

            // 3. Fallback to CPython 3.14 module call (`math.hypot`, `statistics.mean`, etc.)
            return py::bridge().call_module_fn(ex->obj_name, ex->str_val, arg_vals);
        }
    }
    return rt::NaNVal::null();
}

void BytecodeVM::print_val(rt::NaNVal v) {
    std::cout << val_to_string(v);
}

} // namespace flawn
