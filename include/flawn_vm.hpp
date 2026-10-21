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
    std::string type_name; // Empty for generic dict, or struct name
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
    std::unordered_map<std::string, const Stmt*> structs_;

    bool has_returned_ = false;
    bool has_break_ = false;
    bool has_continue_ = false;
    rt::NaNVal return_val_ = rt::NaNVal::null();

    rt::NaNVal make_string(const std::string& s);
    rt::NaNVal make_map(const std::string& type_name = "");
    void exec_stmt(const Stmt* st, std::unordered_map<std::string, rt::NaNVal>& env);
    rt::NaNVal eval_expr(const Expr* ex, std::unordered_map<std::string, rt::NaNVal>& env);
    void print_val(rt::NaNVal v);
};

} // namespace flawn
