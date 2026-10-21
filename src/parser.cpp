#include "flawn_parser.hpp"
#include <stdexcept>

namespace flawn {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

const Token& Parser::peek(size_t offset) const noexcept {
    if (pos_ + offset >= tokens_.size()) return tokens_.back();
    return tokens_[pos_ + offset];
}

const Token& Parser::advance() noexcept {
    if (pos_ >= tokens_.size()) return tokens_.back();
    return tokens_[pos_++];
}

bool Parser::check(TokenKind k) const noexcept {
    return peek().kind == k;
}

bool Parser::match(TokenKind k) noexcept {
    if (check(k)) {
        advance();
        return true;
    }
    return false;
}

void Parser::skip_newlines() noexcept {
    while (check(TokenKind::Newline) || check(TokenKind::Semicolon)) {
        advance();
    }
}

void Parser::consume_stmt_end() noexcept {
    while (check(TokenKind::Semicolon) || check(TokenKind::Newline)) {
        advance();
    }
}

ValueType Parser::parse_type_annotation(std::string* out_struct_name) {
    if (check(TokenKind::KwNull)) {
        advance();
        return ValueType::Void;
    }
    if (!check(TokenKind::Ident)) return ValueType::Inferred;
    std::string tname = advance().text;
    bool is_slice = false;
    if (match(TokenKind::LBracket)) {
        if (!check(TokenKind::RBracket)) {
            // e.g. list[f32] or dict[str, any]
            while (!check(TokenKind::RBracket) && !check(TokenKind::Eof)) advance();
        }
        match(TokenKind::RBracket);
        is_slice = true;
    }
    if (tname == "f32" || tname == "float32") return is_slice ? ValueType::SliceF32 : ValueType::F32;
    if (tname == "f64" || tname == "float" || tname == "double" || tname == "number") return is_slice ? ValueType::SliceF64 : ValueType::F64;
    if (tname == "i64" || tname == "int" || tname == "int64") return is_slice ? ValueType::SliceI64 : ValueType::I64;
    if (tname == "i32" || tname == "int32") return ValueType::I32;
    if (tname == "usize" || tname == "size_t" || tname == "u64") return ValueType::Usize;
    if (tname == "bool" || tname == "boolean") return ValueType::Bool;
    if (tname == "str" || tname == "string") return ValueType::String;
    if (tname == "dict" || tname == "map" || tname == "object") return ValueType::Map;
    if (tname == "list" || tname == "array") return ValueType::SliceF64;
    if (tname == "void" || tname == "None") return ValueType::Void;
    if (struct_names_.count(tname)) {
        if (out_struct_name) *out_struct_name = tname;
        return ValueType::Struct;
    }
    return ValueType::Inferred;
}

std::vector<StmtPtr> Parser::parse_block() {
    std::vector<StmtPtr> body;
    skip_newlines();

    // Mode 1: C++/JS Brace-delimited block `{ ... }`
    if (match(TokenKind::LBrace)) {
        skip_newlines();
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            if (check(TokenKind::Indent) || check(TokenKind::Dedent)) {
                advance();
                continue;
            }
            auto st = parse_stmt();
            if (st) body.push_back(std::move(st));
            skip_newlines();
        }
        match(TokenKind::RBrace);
        skip_newlines();
        return body;
    }

    // Mode 2: Python Colon + Indentation block `: \n INDENT ... DEDENT`
    match(TokenKind::Colon);
    if (check(TokenKind::Newline)) {
        skip_newlines();
        if (match(TokenKind::Indent)) {
            while (!check(TokenKind::Dedent) && !check(TokenKind::Eof)) {
                skip_newlines();
                if (check(TokenKind::Dedent) || check(TokenKind::Eof)) break;
                auto st = parse_stmt();
                if (st) body.push_back(std::move(st));
                skip_newlines();
            }
            match(TokenKind::Dedent);
            skip_newlines();
            return body;
        }
    }

    // Mode 3: Single-line statement block
    auto single = parse_stmt();
    if (single) body.push_back(std::move(single));
    return body;
}

Program Parser::parse_program() {
    Program prog;
    skip_newlines();
    while (!check(TokenKind::Eof)) {
        if (check(TokenKind::Indent) || check(TokenKind::Dedent)) {
            advance();
            continue;
        }
        auto st = parse_stmt();
        if (st) {
            if (st->kind == StmtKind::Import) {
                prog.imports.push_back(st->name);
            }
            prog.statements.push_back(std::move(st));
        }
        skip_newlines();
    }
    return prog;
}

StmtPtr Parser::parse_stmt() {
    skip_newlines();
    if (check(TokenKind::Eof)) return nullptr;

    // Collect hardware optimization decorators (@simd, @parallel, @fastmath, @autotune)
    bool attr_simd = false;
    bool attr_par = false;
    bool attr_fast = false;
    bool attr_tune = false;
    while (check(TokenKind::AtSimd) || check(TokenKind::AtParallel) ||
           check(TokenKind::AtFastmath) || check(TokenKind::AtAutotune) ||
           check(TokenKind::AtDecorator) || check(TokenKind::Newline)) {
        if (match(TokenKind::AtSimd)) attr_simd = true;
        else if (match(TokenKind::AtParallel)) attr_par = true;
        else if (match(TokenKind::AtFastmath)) attr_fast = true;
        else if (match(TokenKind::AtAutotune)) attr_tune = true;
        else if (match(TokenKind::AtDecorator)) {}
        else if (check(TokenKind::Newline)) {
            if (!attr_simd && !attr_par && !attr_fast && !attr_tune) break;
            advance();
        }
    }

    if (match(TokenKind::KwStruct)) {
        return parse_struct_decl();
    }
    if (match(TokenKind::KwFn)) {
        return parse_func_decl(false, attr_simd, attr_par, attr_fast, attr_tune);
    }
    if (match(TokenKind::KwDef)) {
        return parse_func_decl(true, attr_simd, attr_par, attr_fast, attr_tune);
    }
    if (match(TokenKind::KwLet) || match(TokenKind::KwVar)) {
        return parse_var_decl(false);
    }
    if (match(TokenKind::KwConst)) {
        return parse_var_decl(true);
    }
    if (match(TokenKind::KwIf)) {
        return parse_if_stmt();
    }
    if (match(TokenKind::KwFor)) {
        auto fs = parse_for_stmt();
        if (fs) {
            fs->attr_simd |= attr_simd;
            fs->attr_parallel |= attr_par;
        }
        return fs;
    }
    if (match(TokenKind::KwWhile)) {
        return parse_while_stmt();
    }
    if (match(TokenKind::KwBreak)) {
        int ln = peek().line;
        consume_stmt_end();
        return std::make_unique<Stmt>(StmtKind::Break, ln);
    }
    if (match(TokenKind::KwContinue)) {
        int ln = peek().line;
        consume_stmt_end();
        return std::make_unique<Stmt>(StmtKind::Continue, ln);
    }
    if (match(TokenKind::KwAssert)) {
        return parse_assert_stmt();
    }
    if (match(TokenKind::KwReturn)) {
        return parse_return_stmt();
    }
    if (match(TokenKind::KwImport) || match(TokenKind::KwFrom)) {
        return parse_import_stmt();
    }
    if (match(TokenKind::KwPass)) {
        consume_stmt_end();
        return nullptr;
    }

    return parse_ident_or_expr_stmt();
}

StmtPtr Parser::parse_struct_decl() {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::StructDecl, ln);
    if (check(TokenKind::Ident)) {
        st->name = advance().text;
        struct_names_.insert(st->name);
    }
    skip_newlines();
    if (match(TokenKind::LBrace)) {
        skip_newlines();
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            if (check(TokenKind::Indent) || check(TokenKind::Dedent)) { advance(); continue; }
            if (check(TokenKind::Ident)) {
                StructField sf;
                sf.name = advance().text;
                if (match(TokenKind::Colon)) {
                    sf.type = parse_type_annotation();
                }
                if (match(TokenKind::Assign)) {
                    sf.default_val = parse_expr();
                }
                st->fields.push_back(std::move(sf));
                match(TokenKind::Comma);
                consume_stmt_end();
            } else {
                advance();
            }
            skip_newlines();
        }
        match(TokenKind::RBrace);
        skip_newlines();
    } else if (match(TokenKind::Colon)) {
        skip_newlines();
        if (match(TokenKind::Indent)) {
            while (!check(TokenKind::Dedent) && !check(TokenKind::Eof)) {
                skip_newlines();
                if (check(TokenKind::Dedent) || check(TokenKind::Eof)) break;
                if (match(TokenKind::KwPass)) {
                    consume_stmt_end();
                    continue;
                }
                if (check(TokenKind::Ident)) {
                    StructField sf;
                    sf.name = advance().text;
                    if (match(TokenKind::Colon)) {
                        sf.type = parse_type_annotation();
                    }
                    if (match(TokenKind::Assign)) {
                        sf.default_val = parse_expr();
                    }
                    st->fields.push_back(std::move(sf));
                    consume_stmt_end();
                } else {
                    advance();
                }
            }
            match(TokenKind::Dedent);
        }
    }
    return st;
}

StmtPtr Parser::parse_func_decl(bool is_py_def, bool simd, bool par, bool fastmath, bool autotune) {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::FuncDecl, ln);
    st->is_python_def = is_py_def;
    st->attr_simd = simd;
    st->attr_parallel = par;
    st->attr_fastmath = fastmath;
    st->attr_autotune = autotune;

    if (check(TokenKind::Ident)) {
        st->name = advance().text;
    }
    match(TokenKind::LParen);
    while (!check(TokenKind::RParen) && !check(TokenKind::Eof)) {
        Param p;
        if (check(TokenKind::Ident)) {
            p.name = advance().text;
        }
        if (match(TokenKind::Colon)) {
            p.type = parse_type_annotation(&p.struct_type_name);
        }
        if (match(TokenKind::Assign)) {
            p.default_val = parse_expr();
        }
        st->params.push_back(std::move(p));
        if (!match(TokenKind::Comma)) break;
    }
    match(TokenKind::RParen);

    if (match(TokenKind::Arrow) || (!is_py_def && match(TokenKind::Colon))) {
        st->return_type = parse_type_annotation(&st->struct_type_name);
    }

    // Support concise single-expression arrow functions: `fn square(x: f64) => x * x;`
    if (match(TokenKind::FatArrow)) {
        auto ret = std::make_unique<Stmt>(StmtKind::Return, ln);
        ret->expr = parse_expr();
        consume_stmt_end();
        st->body.push_back(std::move(ret));
        return st;
    }

    st->body = parse_block();
    return st;
}

StmtPtr Parser::parse_var_decl(bool is_const) {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::VarDecl, ln);
    st->is_const = is_const;
    if (check(TokenKind::Ident)) {
        st->name = advance().text;
    }
    if (match(TokenKind::Colon)) {
        st->decl_type = parse_type_annotation(&st->struct_type_name);
    }
    if (match(TokenKind::Assign)) {
        st->expr = parse_expr();
    }
    consume_stmt_end();
    return st;
}

StmtPtr Parser::parse_if_stmt() {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::If, ln);
    bool has_paren = match(TokenKind::LParen);
    st->expr = parse_expr();
    if (has_paren) match(TokenKind::RParen);
    st->body = parse_block();

    skip_newlines();
    if (match(TokenKind::KwElif)) {
        st->else_body.push_back(parse_if_stmt());
    } else if (match(TokenKind::KwElse)) {
        if (match(TokenKind::KwIf)) {
            st->else_body.push_back(parse_if_stmt());
        } else {
            st->else_body = parse_block();
        }
    }
    return st;
}

StmtPtr Parser::parse_for_stmt() {
    int ln = peek().line;
    bool has_paren = match(TokenKind::LParen);

    // Check if C++/JS style `for (let i = 0; i < n; i++)` vs Python/Flawn `for i in ...`
    if (check(TokenKind::KwLet) || check(TokenKind::KwVar) ||
        (check(TokenKind::Ident) && peek(1).kind != TokenKind::KwIn)) {
        auto st = std::make_unique<Stmt>(StmtKind::ForCStyle, ln);
        if (match(TokenKind::KwLet) || match(TokenKind::KwVar)) {
            st->c_init = parse_var_decl(false);
        } else {
            st->c_init = parse_ident_or_expr_stmt();
        }
        st->c_cond = parse_expr();
        match(TokenKind::Semicolon);
        st->c_step = parse_ident_or_expr_stmt();
        if (has_paren) match(TokenKind::RParen);
        st->body = parse_block();
        return st;
    }

    std::string lvar;
    if (check(TokenKind::Ident)) {
        lvar = advance().text;
    }
    match(TokenKind::KwIn);

    // Case 1: `for i in range(...)`
    if (check(TokenKind::Ident) && peek().text == "range") {
        auto st = std::make_unique<Stmt>(StmtKind::ForRange, ln);
        st->loop_var = lvar;
        advance(); // consume 'range'
        match(TokenKind::LParen);
        auto first = parse_expr();
        if (match(TokenKind::Comma)) {
            st->range_start = std::move(first);
            st->range_stop = parse_expr();
            if (match(TokenKind::Comma)) {
                st->range_step = parse_expr();
            } else {
                auto step1 = std::make_unique<Expr>(ExprKind::IntLit, ln);
                step1->int_val = 1;
                st->range_step = std::move(step1);
            }
        } else {
            auto start0 = std::make_unique<Expr>(ExprKind::IntLit, ln);
            start0->int_val = 0;
            st->range_start = std::move(start0);
            st->range_stop = std::move(first);
            auto step1 = std::make_unique<Expr>(ExprKind::IntLit, ln);
            step1->int_val = 1;
            st->range_step = std::move(step1);
        }
        match(TokenKind::RParen);
        if (has_paren) match(TokenKind::RParen);
        st->body = parse_block();
        return st;
    }

    // Parse first expression after `in`
    auto first_expr = parse_expr();

    // Case 2: Ergonomic Range operator `for i in 0..n` or `for i in 0..=n`
    if (check(TokenKind::DotDot) || check(TokenKind::DotDotEq)) {
        bool inclusive = (advance().kind == TokenKind::DotDotEq);
        auto stop_expr = parse_expr();
        auto st = std::make_unique<Stmt>(StmtKind::ForRange, ln);
        st->loop_var = lvar;
        st->range_start = std::move(first_expr);
        if (inclusive) {
            auto plus1 = std::make_unique<Expr>(ExprKind::Binary, ln);
            plus1->str_val = "+";
            plus1->lhs = std::move(stop_expr);
            auto one = std::make_unique<Expr>(ExprKind::IntLit, ln);
            one->int_val = 1;
            plus1->rhs = std::move(one);
            st->range_stop = std::move(plus1);
        } else {
            st->range_stop = std::move(stop_expr);
        }
        auto step1 = std::make_unique<Expr>(ExprKind::IntLit, ln);
        step1->int_val = 1;
        st->range_step = std::move(step1);
        if (has_paren) match(TokenKind::RParen);
        st->body = parse_block();
        return st;
    }

    // Case 3: `for item in collection` (ForEach)
    auto st = std::make_unique<Stmt>(StmtKind::ForEach, ln);
    st->loop_var = lvar;
    st->expr = std::move(first_expr);
    if (has_paren) match(TokenKind::RParen);
    st->body = parse_block();
    return st;
}

StmtPtr Parser::parse_while_stmt() {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::While, ln);
    bool has_paren = match(TokenKind::LParen);
    st->expr = parse_expr();
    if (has_paren) match(TokenKind::RParen);
    st->body = parse_block();
    return st;
}

StmtPtr Parser::parse_assert_stmt() {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::Assert, ln);
    bool has_paren = match(TokenKind::LParen);
    st->expr = parse_expr();
    if (match(TokenKind::Comma)) {
        st->assert_msg = parse_expr();
    }
    if (has_paren) match(TokenKind::RParen);
    consume_stmt_end();
    return st;
}

StmtPtr Parser::parse_return_stmt() {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::Return, ln);
    if (!check(TokenKind::Newline) && !check(TokenKind::Semicolon) &&
        !check(TokenKind::RBrace) && !check(TokenKind::Dedent) && !check(TokenKind::Eof)) {
        st->expr = parse_expr();
    }
    consume_stmt_end();
    return st;
}

StmtPtr Parser::parse_import_stmt() {
    int ln = peek().line;
    auto st = std::make_unique<Stmt>(StmtKind::Import, ln);
    if (check(TokenKind::Ident)) {
        st->name = advance().text;
    }
    while (!check(TokenKind::Newline) && !check(TokenKind::Semicolon) && !check(TokenKind::Eof)) {
        advance();
    }
    consume_stmt_end();
    return st;
}

StmtPtr Parser::parse_ident_or_expr_stmt() {
    int ln = peek().line;

    if (check(TokenKind::Ident)) {
        TokenKind next_k = peek(1).kind;

        // Python-style typed variable declaration `x: i64 = 10`
        if (next_k == TokenKind::Colon) {
            std::string var_name = advance().text;
            advance(); // ':'
            auto st = std::make_unique<Stmt>(StmtKind::VarDecl, ln);
            st->name = var_name;
            st->decl_type = parse_type_annotation(&st->struct_type_name);
            if (match(TokenKind::Assign)) {
                st->expr = parse_expr();
            }
            consume_stmt_end();
            return st;
        }

        // Struct field assignment `obj.field = expr` or `obj.field += expr`
        if (next_k == TokenKind::Dot && peek(2).kind == TokenKind::Ident &&
            (peek(3).kind == TokenKind::Assign || peek(3).kind == TokenKind::PlusEq ||
             peek(3).kind == TokenKind::MinusEq || peek(3).kind == TokenKind::StarEq)) {
            std::string obj_name = advance().text;
            advance(); // '.'
            std::string fld_name = advance().text;
            Token op_tok = advance();
            auto st = std::make_unique<Stmt>(StmtKind::Assign, ln);
            st->name = obj_name;
            st->field_name = fld_name;
            st->assign_op = op_tok.text;
            st->expr = parse_expr();
            consume_stmt_end();
            return st;
        }

        // Variable assignment or compound assignment (`x = ...`, `x += ...`, `i++`)
        if (next_k == TokenKind::Assign || next_k == TokenKind::PlusEq ||
            next_k == TokenKind::MinusEq || next_k == TokenKind::StarEq ||
            next_k == TokenKind::SlashEq || next_k == TokenKind::PlusPlus ||
            next_k == TokenKind::MinusMinus) {
            std::string var_name = advance().text;
            Token op_tok = advance();
            auto st = std::make_unique<Stmt>(StmtKind::Assign, ln);
            st->name = var_name;
            if (op_tok.kind == TokenKind::PlusPlus) {
                st->assign_op = "+=";
                auto one = std::make_unique<Expr>(ExprKind::IntLit, ln);
                one->int_val = 1;
                st->expr = std::move(one);
            } else if (op_tok.kind == TokenKind::MinusMinus) {
                st->assign_op = "-=";
                auto one = std::make_unique<Expr>(ExprKind::IntLit, ln);
                one->int_val = 1;
                st->expr = std::move(one);
            } else {
                st->assign_op = op_tok.text;
                st->expr = parse_expr();
            }
            consume_stmt_end();
            return st;
        }

        // Array / Map indexed assignment `arr[i] = expr` or `map["k"] = expr`
        if (next_k == TokenKind::LBracket) {
            size_t saved_pos = pos_;
            std::string arr_name = advance().text;
            advance(); // '['
            auto idx_expr = parse_expr();
            if (match(TokenKind::RBracket)) {
                if (check(TokenKind::Assign) || check(TokenKind::PlusEq) ||
                    check(TokenKind::MinusEq) || check(TokenKind::StarEq)) {
                    Token op_tok = advance();
                    auto st = std::make_unique<Stmt>(StmtKind::Assign, ln);
                    st->name = arr_name;
                    st->assign_op = op_tok.text;
                    st->index_expr = std::move(idx_expr);
                    st->expr = parse_expr();
                    consume_stmt_end();
                    return st;
                }
            }
            pos_ = saved_pos;
        }
    }

    auto ex = parse_expr();
    consume_stmt_end();
    if (!ex) {
        if (!check(TokenKind::Eof)) advance();
        return nullptr;
    }
    auto st = std::make_unique<Stmt>(StmtKind::ExprStmt, ln);
    st->expr = std::move(ex);
    return st;
}

ExprPtr Parser::parse_expr() { return parse_pipe(); }

// Pipeline operator: `x |> f` or `x |> f(a, b)` -> `f(x)` or `f(x, a, b)`
ExprPtr Parser::parse_pipe() {
    auto lhs = parse_ternary();
    while (match(TokenKind::PipeGt)) {
        int ln = peek().line;
        auto rhs = parse_ternary();
        if (rhs && rhs->kind == ExprKind::Var) {
            auto call = std::make_unique<Expr>(ExprKind::Call, ln);
            call->str_val = rhs->str_val;
            call->args.push_back(std::move(lhs));
            lhs = std::move(call);
        } else if (rhs && rhs->kind == ExprKind::Call) {
            rhs->args.insert(rhs->args.begin(), std::move(lhs));
            lhs = std::move(rhs);
        } else if (rhs && rhs->kind == ExprKind::MemberCall) {
            rhs->args.insert(rhs->args.begin(), std::move(lhs));
            lhs = std::move(rhs);
        }
    }
    return lhs;
}

// Ternary operator: C++/JS `cond ? a : b` and Python `a if cond else b`
ExprPtr Parser::parse_ternary() {
    auto expr = parse_null_coalesce();
    if (match(TokenKind::Question)) {
        int ln = peek().line;
        auto true_br = parse_expr();
        match(TokenKind::Colon);
        auto false_br = parse_ternary();
        auto tern = std::make_unique<Expr>(ExprKind::Ternary, ln);
        tern->cond_expr = std::move(expr);
        tern->lhs = std::move(true_br);
        tern->rhs = std::move(false_br);
        return tern;
    }
    if (match(TokenKind::KwIf)) {
        int ln = peek().line;
        auto cond = parse_null_coalesce();
        match(TokenKind::KwElse);
        auto false_br = parse_ternary();
        auto tern = std::make_unique<Expr>(ExprKind::Ternary, ln);
        tern->cond_expr = std::move(cond);
        tern->lhs = std::move(expr);
        tern->rhs = std::move(false_br);
        return tern;
    }
    return expr;
}

// Null-coalescing operator: `a ?? b`
ExprPtr Parser::parse_null_coalesce() {
    auto lhs = parse_logical_or();
    while (match(TokenKind::NullCoalesce)) {
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = "??";
        bin->lhs = std::move(lhs);
        bin->rhs = parse_logical_or();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_logical_or() {
    auto lhs = parse_logical_and();
    while (match(TokenKind::KwOr)) {
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = "||";
        bin->lhs = std::move(lhs);
        bin->rhs = parse_logical_and();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_logical_and() {
    auto lhs = parse_equality();
    while (match(TokenKind::KwAnd)) {
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = "&&";
        bin->lhs = std::move(lhs);
        bin->rhs = parse_equality();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_equality() {
    auto lhs = parse_comparison();
    while (check(TokenKind::Eq) || check(TokenKind::Neq)) {
        std::string op = advance().text;
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = op;
        bin->lhs = std::move(lhs);
        bin->rhs = parse_comparison();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_comparison() {
    auto lhs = parse_bitwise();
    while (check(TokenKind::Lt) || check(TokenKind::Le) ||
           check(TokenKind::Gt) || check(TokenKind::Ge)) {
        std::string op = advance().text;
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = op;
        bin->lhs = std::move(lhs);
        bin->rhs = parse_bitwise();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_bitwise() {
    auto lhs = parse_Shift();
    while (check(TokenKind::Ampersand) || check(TokenKind::Pipe) || check(TokenKind::Caret)) {
        std::string op = advance().text;
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = op;
        bin->lhs = std::move(lhs);
        bin->rhs = parse_Shift();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_Shift() {
    auto lhs = parse_additive();
    while (check(TokenKind::LShift) || check(TokenKind::RShift)) {
        std::string op = advance().text;
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = op;
        bin->lhs = std::move(lhs);
        bin->rhs = parse_additive();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_additive() {
    auto lhs = parse_multiplicative();
    while (check(TokenKind::Plus) || check(TokenKind::Minus)) {
        std::string op = advance().text;
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = op;
        bin->lhs = std::move(lhs);
        bin->rhs = parse_multiplicative();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_multiplicative() {
    auto lhs = parse_unary();
    while (check(TokenKind::Star) || check(TokenKind::Slash) ||
           check(TokenKind::FloorDiv) || check(TokenKind::Percent) ||
           check(TokenKind::Power)) {
        std::string op = advance().text;
        auto bin = std::make_unique<Expr>(ExprKind::Binary, peek().line);
        bin->str_val = op;
        bin->lhs = std::move(lhs);
        bin->rhs = parse_unary();
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parse_unary() {
    if (check(TokenKind::Minus) || check(TokenKind::KwNot)) {
        std::string op = advance().text;
        auto un = std::make_unique<Expr>(ExprKind::Unary, peek().line);
        un->str_val = (op == "not") ? "!" : op;
        un->lhs = parse_unary();
        return un;
    }
    return parse_postfix();
}

ExprPtr Parser::parse_postfix() {
    auto expr = parse_primary();
    while (true) {
        if (match(TokenKind::LParen)) {
            auto call = std::make_unique<Expr>(ExprKind::Call, peek().line);
            if (expr && expr->kind == ExprKind::Var) {
                call->str_val = expr->str_val;
            }
            while (!check(TokenKind::RParen) && !check(TokenKind::Eof)) {
                call->args.push_back(parse_expr());
                if (!match(TokenKind::Comma)) break;
            }
            match(TokenKind::RParen);
            expr = std::move(call);
        } else if (match(TokenKind::Dot)) {
            std::string member = check(TokenKind::Ident) ? advance().text : "";
            if (match(TokenKind::LParen)) {
                auto mcall = std::make_unique<Expr>(ExprKind::MemberCall, peek().line);
                mcall->obj_name = (expr && expr->kind == ExprKind::Var) ? expr->str_val : "";
                mcall->lhs = std::move(expr);
                mcall->str_val = member;
                while (!check(TokenKind::RParen) && !check(TokenKind::Eof)) {
                    mcall->args.push_back(parse_expr());
                    if (!match(TokenKind::Comma)) break;
                }
                match(TokenKind::RParen);
                expr = std::move(mcall);
            } else {
                auto fget = std::make_unique<Expr>(ExprKind::FieldAccess, peek().line);
                fget->obj_name = (expr && expr->kind == ExprKind::Var) ? expr->str_val : "";
                fget->lhs = std::move(expr);
                fget->str_val = member;
                expr = std::move(fget);
            }
        } else if (match(TokenKind::LBracket)) {
            auto idx = parse_expr();
            match(TokenKind::RBracket);
            if (expr && expr->kind == ExprKind::Var &&
                (expr->str_val == "f32" || expr->str_val == "f64" || expr->str_val == "i64")) {
                auto alloc = std::make_unique<Expr>(ExprKind::ArrayAlloc, peek().line);
                if (expr->str_val == "f32") alloc->alloc_elem_type = ValueType::F32;
                else if (expr->str_val == "f64") alloc->alloc_elem_type = ValueType::F64;
                else alloc->alloc_elem_type = ValueType::I64;
                alloc->lhs = std::move(idx);
                expr = std::move(alloc);
            } else {
                auto index_node = std::make_unique<Expr>(ExprKind::Index, peek().line);
                index_node->lhs = std::move(expr);
                index_node->rhs = std::move(idx);
                expr = std::move(index_node);
            }
        } else {
            break;
        }
    }
    return expr;
}

ExprPtr Parser::parse_fstring(const std::string& raw, int ln) {
    auto fexpr = std::make_unique<Expr>(ExprKind::FString, ln);
    fexpr->inferred_type = ValueType::String;
    std::string cur_lit;
    size_t i = 0;
    while (i < raw.size()) {
        if (raw[i] == '{') {
            fexpr->keys.push_back(cur_lit);
            cur_lit.clear();
            i++;
            std::string inner_code;
            int depth = 1;
            while (i < raw.size() && depth > 0) {
                if (raw[i] == '{') depth++;
                else if (raw[i] == '}') {
                    depth--;
                    if (depth == 0) { i++; break; }
                }
                inner_code.push_back(raw[i++]);
            }
            Lexer sub_lex(inner_code);
            Parser sub_par(sub_lex.tokenize());
            auto sub_e = sub_par.parse_expr();
            if (!sub_e) {
                sub_e = std::make_unique<Expr>(ExprKind::StringLit, ln);
            }
            fexpr->args.push_back(std::move(sub_e));
        } else {
            cur_lit.push_back(raw[i++]);
        }
    }
    fexpr->keys.push_back(cur_lit);
    return fexpr;
}

ExprPtr Parser::parse_primary() {
    int ln = peek().line;
    if (check(TokenKind::IntLit)) {
        Token t = advance();
        auto ex = std::make_unique<Expr>(ExprKind::IntLit, ln);
        ex->int_val = t.int_val;
        ex->inferred_type = ValueType::I64;
        return ex;
    }
    if (check(TokenKind::FloatLit)) {
        Token t = advance();
        auto ex = std::make_unique<Expr>(ExprKind::FloatLit, ln);
        ex->float_val = t.float_val;
        ex->inferred_type = ValueType::F64;
        return ex;
    }
    if (match(TokenKind::KwTrue)) {
        auto ex = std::make_unique<Expr>(ExprKind::BoolLit, ln);
        ex->bool_val = true;
        ex->inferred_type = ValueType::Bool;
        return ex;
    }
    if (match(TokenKind::KwFalse)) {
        auto ex = std::make_unique<Expr>(ExprKind::BoolLit, ln);
        ex->bool_val = false;
        ex->inferred_type = ValueType::Bool;
        return ex;
    }
    if (match(TokenKind::KwNull)) {
        auto ex = std::make_unique<Expr>(ExprKind::NullLit, ln);
        ex->inferred_type = ValueType::Void;
        return ex;
    }
    if (check(TokenKind::StringLit)) {
        Token t = advance();
        auto ex = std::make_unique<Expr>(ExprKind::StringLit, ln);
        ex->str_val = t.text;
        ex->inferred_type = ValueType::String;
        return ex;
    }
    if (check(TokenKind::FStringLit)) {
        Token t = advance();
        return parse_fstring(t.text, ln);
    }
    if (check(TokenKind::Ident)) {
        Token t = advance();
        // Check for Struct Instantiation `Point { x: 1.0, y: 2.0 }`
        if (struct_names_.count(t.text) && check(TokenKind::LBrace)) {
            advance(); // '{'
            auto sinit = std::make_unique<Expr>(ExprKind::StructInit, ln);
            sinit->struct_type_name = t.text;
            sinit->inferred_type = ValueType::Struct;
            skip_newlines();
            while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
                if (check(TokenKind::Ident) && peek(1).kind == TokenKind::Colon) {
                    sinit->keys.push_back(advance().text);
                    advance(); // ':'
                    sinit->args.push_back(parse_expr());
                } else {
                    sinit->keys.push_back("");
                    sinit->args.push_back(parse_expr());
                }
                match(TokenKind::Comma);
                skip_newlines();
            }
            match(TokenKind::RBrace);
            return sinit;
        }
        auto ex = std::make_unique<Expr>(ExprKind::Var, ln);
        ex->str_val = t.text;
        return ex;
    }
    if (match(TokenKind::LParen)) {
        auto inner = parse_expr();
        match(TokenKind::RParen);
        return inner;
    }
    // Array Literals `[1, 2, 3, 4]` or Python repeat `[0.0] * n`
    if (match(TokenKind::LBracket)) {
        std::vector<ExprPtr> elems;
        while (!check(TokenKind::RBracket) && !check(TokenKind::Eof)) {
            elems.push_back(parse_expr());
            if (!match(TokenKind::Comma)) break;
        }
        match(TokenKind::RBracket);
        if (elems.size() == 1 && match(TokenKind::Star)) {
            auto count = parse_expr();
            auto alloc = std::make_unique<Expr>(ExprKind::ArrayAlloc, ln);
            alloc->alloc_elem_type = (elems[0] && elems[0]->kind == ExprKind::IntLit) ? ValueType::I64 : ValueType::F32;
            alloc->lhs = std::move(count);
            return alloc;
        }
        auto alit = std::make_unique<Expr>(ExprKind::ArrayLit, ln);
        alit->args = std::move(elems);
        alit->inferred_type = ValueType::SliceF64;
        return alit;
    }
    // Dictionary / Map Literal `{ "host": "localhost", "port": 8080 }`
    if (match(TokenKind::LBrace)) {
        auto mlit = std::make_unique<Expr>(ExprKind::MapLit, ln);
        mlit->inferred_type = ValueType::Map;
        skip_newlines();
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            std::string key;
            if (check(TokenKind::StringLit) || check(TokenKind::Ident)) {
                key = advance().text;
            }
            match(TokenKind::Colon);
            mlit->keys.push_back(key);
            mlit->args.push_back(parse_expr());
            match(TokenKind::Comma);
            skip_newlines();
        }
        match(TokenKind::RBrace);
        return mlit;
    }
    return nullptr;
}

} // namespace flawn
