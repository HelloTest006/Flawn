#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace flawn {

enum class TokenKind {
    Eof,
    Newline,
    Indent,
    Dedent,

    // Identifiers & Literals
    Ident,
    IntLit,
    FloatLit,
    StringLit,
    FStringLit, // f"..." or `...${expr}...`

    // Dual-Grammar Keywords (Flawn C++/JS + Python)
    KwFn,       // fn / function
    KwDef,      // def
    KwLet,      // let
    KwConst,    // const
    KwVar,      // var
    KwStruct,   // struct / class
    KwReturn,   // return
    KwIf,       // if
    KwElif,     // elif
    KwElse,     // else
    KwFor,      // for
    KwIn,       // in
    KwWhile,    // while
    KwBreak,    // break
    KwContinue, // continue
    KwAssert,   // assert
    KwImport,   // import
    KwFrom,     // from
    KwAs,       // as
    KwPass,     // pass
    KwTrue,     // true / True
    KwFalse,    // false / False
    KwNull,     // null / None / nil
    KwAnd,      // and / &&
    KwOr,       // or / ||
    KwNot,      // not / !

    // Decorators / Hardware Optimization Annotations
    AtSimd,     // @simd
    AtParallel, // @parallel
    AtFastmath, // @fastmath
    AtAutotune, // @autotune
    AtDecorator,// generic @decorator

    // Operators & Punctuation
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Comma,      // ,
    Colon,      // :
    Semicolon,  // ;
    Dot,        // .
    DotDot,     // ..
    DotDotEq,   // ..=
    Arrow,      // ->
    FatArrow,   // =>
    PipeGt,     // |>
    NullCoalesce,// ??
    Question,   // ?

    Plus,       // +
    Minus,      // -
    Star,       // *
    Power,      // **
    Slash,      // /
    FloorDiv,   // //
    Percent,    // %
    PlusEq,     // +=
    MinusEq,    // -=
    StarEq,     // *=
    SlashEq,    // /=
    PlusPlus,   // ++
    MinusMinus, // --

    Assign,     // =
    Eq,         // ==
    Neq,        // !=
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    Ampersand,  // &
    Pipe,       // |
    Caret,      // ^
    LShift,     // <<
    RShift      // >>
};

struct Token {
    TokenKind kind;
    std::string text;
    int64_t int_val = 0;
    double float_val = 0.0;
    int line = 1;
    int col = 1;
};

class Lexer {
public:
    explicit Lexer(std::string source);
    std::vector<Token> tokenize();

private:
    std::string src_;
    size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;
    int brace_depth_ = 0;
    int paren_depth_ = 0;
    std::vector<int> indent_stack_{0};

    char peek(size_t offset = 0) const noexcept;
    char advance() noexcept;
    bool is_eof() const noexcept;
};

} // namespace flawn
