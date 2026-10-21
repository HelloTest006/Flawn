#pragma once
#include "flawn_lexer.hpp"
#include "flawn_ast.hpp"
#include <vector>
#include <unordered_set>

namespace flawn {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);
    Program parse_program();

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;
    std::unordered_set<std::string> struct_names_;

    const Token& peek(size_t offset = 0) const noexcept;
    const Token& advance() noexcept;
    bool check(TokenKind k) const noexcept;
    bool match(TokenKind k) noexcept;
    void skip_newlines() noexcept;
    void consume_stmt_end() noexcept;

    ValueType parse_type_annotation(std::string* out_struct_name = nullptr);
    std::vector<StmtPtr> parse_block();

    StmtPtr parse_stmt();
    StmtPtr parse_struct_decl();
    StmtPtr parse_func_decl(bool is_py_def, bool simd, bool par, bool fastmath, bool autotune);
    StmtPtr parse_var_decl(bool is_const);
    StmtPtr parse_if_stmt();
    StmtPtr parse_for_stmt();
    StmtPtr parse_while_stmt();
    StmtPtr parse_assert_stmt();
    StmtPtr parse_return_stmt();
    StmtPtr parse_import_stmt();
    StmtPtr parse_ident_or_expr_stmt();

    ExprPtr parse_expr();
    ExprPtr parse_pipe();
    ExprPtr parse_ternary();
    ExprPtr parse_null_coalesce();
    ExprPtr parse_logical_or();
    ExprPtr parse_logical_and();
    ExprPtr parse_equality();
    ExprPtr parse_comparison();
    ExprPtr parse_bitwise();
    ExprPtr parse_Shift();
    ExprPtr parse_additive();
    ExprPtr parse_multiplicative();
    ExprPtr parse_unary();
    ExprPtr parse_postfix();
    ExprPtr parse_primary();
    ExprPtr parse_fstring(const std::string& raw_fstr, int ln);
};

} // namespace flawn
