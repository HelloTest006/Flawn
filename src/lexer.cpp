#include "flawn_lexer.hpp"
#include <cctype>
#include <cstdlib>

namespace flawn {

Lexer::Lexer(std::string source) : src_(std::move(source)) {}

char Lexer::peek(size_t offset) const noexcept {
    if (pos_ + offset >= src_.size()) return '\0';
    return src_[pos_ + offset];
}

char Lexer::advance() noexcept {
    if (pos_ >= src_.size()) return '\0';
    char c = src_[pos_++];
    if (c == '\n') {
        line_++;
        col_ = 1;
    } else {
        col_++;
    }
    return c;
}

bool Lexer::is_eof() const noexcept {
    return pos_ >= src_.size();
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    bool at_line_start = true;

    while (!is_eof()) {
        // Handle Python-style indentation at start of logical line when not inside { } or ( )
        if (at_line_start && brace_depth_ == 0 && paren_depth_ == 0) {
            int indent = 0;
            while (!is_eof()) {
                char c = peek();
                if (c == ' ') {
                    indent++;
                    advance();
                } else if (c == '\t') {
                    indent += 4;
                    advance();
                } else if (c == '\r') {
                    advance();
                } else {
                    break;
                }
            }

            // Skip blank lines or comment-only lines without changing indent stack
            char c = peek();
            if (c == '\n') {
                advance();
                at_line_start = true;
                continue;
            }
            if (c == '#' || (c == '/' && peek(1) == '/')) {
                while (!is_eof() && peek() != '\n') advance();
                if (!is_eof() && peek() == '\n') advance();
                at_line_start = true;
                continue;
            }
            if (is_eof()) break;

            int current_indent = indent_stack_.back();
            if (indent > current_indent) {
                indent_stack_.push_back(indent);
                tokens.push_back(Token{TokenKind::Indent, "<INDENT>", 0, 0.0, line_, col_});
            } else if (indent < current_indent) {
                while (indent_stack_.size() > 1 && indent_stack_.back() > indent) {
                    indent_stack_.pop_back();
                    tokens.push_back(Token{TokenKind::Dedent, "<DEDENT>", 0, 0.0, line_, col_});
                }
            }
            at_line_start = false;
        }

        char c = peek();
        if (c == '\0') break;

        // Horizontal whitespace
        if (c == ' ' || c == '\t' || c == '\r') {
            advance();
            continue;
        }

        // Comments: Python `#` or C++/JS `//` only at start of line or after `}` / `;`
        if (c == '#' || (c == '/' && peek(1) == '/' &&
            (tokens.empty() || tokens.back().kind == TokenKind::Newline ||
             tokens.back().kind == TokenKind::Semicolon || tokens.back().kind == TokenKind::RBrace ||
             tokens.back().kind == TokenKind::LBrace))) {
            while (!is_eof() && peek() != '\n') advance();
            continue;
        }
        if (c == '/' && peek(1) == '*') {
            advance();
            advance();
            while (!is_eof() && !(peek() == '*' && peek(1) == '/')) advance();
            if (!is_eof()) {
                advance();
                advance();
            }
            continue;
        }

        // Newline
        if (c == '\n') {
            int ln = line_, cl = col_;
            advance();
            if (paren_depth_ == 0 && brace_depth_ == 0) {
                if (!tokens.empty() && tokens.back().kind != TokenKind::Newline &&
                    tokens.back().kind != TokenKind::Indent) {
                    tokens.push_back(Token{TokenKind::Newline, "\\n", 0, 0.0, ln, cl});
                }
                at_line_start = true;
            }
            continue;
        }

        int tok_line = line_;
        int tok_col = col_;

        // Decorators / Hardware Optimization Annotations (`@simd`, `@parallel`, `@fastmath`, `@autotune`)
        if (c == '@') {
            advance();
            std::string name;
            while (!is_eof() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')) {
                name.push_back(advance());
            }
            TokenKind kind = TokenKind::AtDecorator;
            if (name == "simd") kind = TokenKind::AtSimd;
            else if (name == "parallel" || name == "par") kind = TokenKind::AtParallel;
            else if (name == "fastmath") kind = TokenKind::AtFastmath;
            else if (name == "autotune") kind = TokenKind::AtAutotune;
            tokens.push_back(Token{kind, "@" + name, 0, 0.0, tok_line, tok_col});
            continue;
        }

        // Python F-String `f"..."` or `f'...'`
        if ((c == 'f' || c == 'F') && (peek(1) == '"' || peek(1) == '\'')) {
            advance(); // 'f'
            char quote = advance();
            std::string val;
            while (!is_eof() && peek() != quote) {
                if (peek() == '\\') {
                    advance();
                    char esc = advance();
                    switch (esc) {
                        case 'n': val.push_back('\n'); break;
                        case 't': val.push_back('\t'); break;
                        case 'r': val.push_back('\r'); break;
                        default: val.push_back(esc); break;
                    }
                } else {
                    val.push_back(advance());
                }
            }
            if (!is_eof() && peek() == quote) advance();
            tokens.push_back(Token{TokenKind::FStringLit, val, 0, 0.0, tok_line, tok_col});
            continue;
        }

        // JavaScript Backtick Template Literal `` `...${expr}...` ``
        if (c == '`') {
            advance();
            std::string val;
            while (!is_eof() && peek() != '`') {
                if (peek() == '$' && peek(1) == '{') {
                    advance(); // consume '$' so `{expr}` matches unified FString format
                    val.push_back(advance()); // '{'
                } else if (peek() == '\\') {
                    advance();
                    char esc = advance();
                    switch (esc) {
                        case 'n': val.push_back('\n'); break;
                        case 't': val.push_back('\t'); break;
                        default: val.push_back(esc); break;
                    }
                } else {
                    val.push_back(advance());
                }
            }
            if (!is_eof() && peek() == '`') advance();
            tokens.push_back(Token{TokenKind::FStringLit, val, 0, 0.0, tok_line, tok_col});
            continue;
        }

        // String Literals ('...' or "...")
        if (c == '"' || c == '\'') {
            char quote = advance();
            std::string val;
            while (!is_eof() && peek() != quote) {
                if (peek() == '\\') {
                    advance();
                    char esc = advance();
                    switch (esc) {
                        case 'n': val.push_back('\n'); break;
                        case 't': val.push_back('\t'); break;
                        case 'r': val.push_back('\r'); break;
                        case '\\': val.push_back('\\'); break;
                        case '"': val.push_back('"'); break;
                        case '\'': val.push_back('\''); break;
                        default: val.push_back(esc); break;
                    }
                } else {
                    val.push_back(advance());
                }
            }
            if (!is_eof() && peek() == quote) advance();
            tokens.push_back(Token{TokenKind::StringLit, val, 0, 0.0, tok_line, tok_col});
            continue;
        }

        // Numeric Literals (Hex 0x, Binary 0b, Integers & Floats, supports `_` separators)
        if (std::isdigit(static_cast<unsigned char>(c))) {
            if (c == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
                advance(); advance();
                std::string hex_s;
                while (!is_eof() && (std::isxdigit(static_cast<unsigned char>(peek())) || peek() == '_')) {
                    char hc = advance();
                    if (hc != '_') hex_s.push_back(hc);
                }
                int64_t iv = std::strtoll(hex_s.c_str(), nullptr, 16);
                tokens.push_back(Token{TokenKind::IntLit, hex_s, iv, static_cast<double>(iv), tok_line, tok_col});
                continue;
            }
            if (c == '0' && (peek(1) == 'b' || peek(1) == 'B')) {
                advance(); advance();
                std::string bin_s;
                while (!is_eof() && (peek() == '0' || peek() == '1' || peek() == '_')) {
                    char bc = advance();
                    if (bc != '_') bin_s.push_back(bc);
                }
                int64_t iv = std::strtoll(bin_s.c_str(), nullptr, 2);
                tokens.push_back(Token{TokenKind::IntLit, bin_s, iv, static_cast<double>(iv), tok_line, tok_col});
                continue;
            }

            std::string num_str;
            bool is_float = false;
            while (!is_eof()) {
                char nc = peek();
                if (std::isdigit(static_cast<unsigned char>(nc))) {
                    num_str.push_back(advance());
                } else if (nc == '_') {
                    advance();
                } else if (nc == '.' && peek(1) != '.' && !is_float && std::isdigit(static_cast<unsigned char>(peek(1)))) {
                    is_float = true;
                    num_str.push_back(advance());
                } else if ((nc == 'e' || nc == 'E') && !num_str.empty()) {
                    is_float = true;
                    num_str.push_back(advance());
                    if (peek() == '+' || peek() == '-') num_str.push_back(advance());
                } else {
                    break;
                }
            }
            // Optional type suffix like `f32`, `f64`, `i64`
            if (peek() == 'f' || peek() == 'i' || peek() == 'u') {
                while (std::isalnum(static_cast<unsigned char>(peek()))) advance();
            }
            if (is_float) {
                double fv = std::strtod(num_str.c_str(), nullptr);
                tokens.push_back(Token{TokenKind::FloatLit, num_str, 0, fv, tok_line, tok_col});
            } else {
                int64_t iv = std::strtoll(num_str.c_str(), nullptr, 10);
                tokens.push_back(Token{TokenKind::IntLit, num_str, iv, static_cast<double>(iv), tok_line, tok_col});
            }
            continue;
        }

        // Identifiers & Keywords
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::string id;
            while (!is_eof() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')) {
                id.push_back(advance());
            }
            TokenKind kind = TokenKind::Ident;
            if (id == "fn" || id == "function") kind = TokenKind::KwFn;
            else if (id == "def") kind = TokenKind::KwDef;
            else if (id == "let") kind = TokenKind::KwLet;
            else if (id == "const") kind = TokenKind::KwConst;
            else if (id == "var") kind = TokenKind::KwVar;
            else if (id == "struct" || id == "class") kind = TokenKind::KwStruct;
            else if (id == "return") kind = TokenKind::KwReturn;
            else if (id == "if") kind = TokenKind::KwIf;
            else if (id == "elif") kind = TokenKind::KwElif;
            else if (id == "else") kind = TokenKind::KwElse;
            else if (id == "for") kind = TokenKind::KwFor;
            else if (id == "in") kind = TokenKind::KwIn;
            else if (id == "while") kind = TokenKind::KwWhile;
            else if (id == "break") kind = TokenKind::KwBreak;
            else if (id == "continue") kind = TokenKind::KwContinue;
            else if (id == "assert") kind = TokenKind::KwAssert;
            else if (id == "import") kind = TokenKind::KwImport;
            else if (id == "from") kind = TokenKind::KwFrom;
            else if (id == "as") kind = TokenKind::KwAs;
            else if (id == "pass") kind = TokenKind::KwPass;
            else if (id == "true" || id == "True") kind = TokenKind::KwTrue;
            else if (id == "false" || id == "False") kind = TokenKind::KwFalse;
            else if (id == "null" || id == "None" || id == "nil") kind = TokenKind::KwNull;
            else if (id == "and") kind = TokenKind::KwAnd;
            else if (id == "or") kind = TokenKind::KwOr;
            else if (id == "not") kind = TokenKind::KwNot;

            tokens.push_back(Token{kind, id, 0, 0.0, tok_line, tok_col});
            continue;
        }

        // Multi-character operators
        char c1 = advance();
        char c2 = peek();

        if (c1 == '.' && c2 == '.') {
            advance();
            if (peek() == '=') {
                advance();
                tokens.push_back(Token{TokenKind::DotDotEq, "..=", 0, 0.0, tok_line, tok_col});
            } else {
                tokens.push_back(Token{TokenKind::DotDot, "..", 0, 0.0, tok_line, tok_col});
            }
            continue;
        }
        if (c1 == '|' && c2 == '>') { advance(); tokens.push_back(Token{TokenKind::PipeGt, "|>", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '?' && c2 == '?') { advance(); tokens.push_back(Token{TokenKind::NullCoalesce, "??", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '=' && c2 == '>') { advance(); tokens.push_back(Token{TokenKind::FatArrow, "=>", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '-' && c2 == '>') { advance(); tokens.push_back(Token{TokenKind::Arrow, "->", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '+' && c2 == '+') { advance(); tokens.push_back(Token{TokenKind::PlusPlus, "++", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '-' && c2 == '-') { advance(); tokens.push_back(Token{TokenKind::MinusMinus, "--", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '+' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::PlusEq, "+=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '-' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::MinusEq, "-=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '*' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::StarEq, "*=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '/' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::SlashEq, "/=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '*' && c2 == '*') { advance(); tokens.push_back(Token{TokenKind::Power, "**", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '/' && c2 == '/') { advance(); tokens.push_back(Token{TokenKind::FloorDiv, "//", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '=' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::Eq, "==", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '!' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::Neq, "!=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '<' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::Le, "<=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '>' && c2 == '=') { advance(); tokens.push_back(Token{TokenKind::Ge, ">=", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '<' && c2 == '<') { advance(); tokens.push_back(Token{TokenKind::LShift, "<<", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '>' && c2 == '>') { advance(); tokens.push_back(Token{TokenKind::RShift, ">>", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '&' && c2 == '&') { advance(); tokens.push_back(Token{TokenKind::KwAnd, "&&", 0, 0.0, tok_line, tok_col}); continue; }
        if (c1 == '|' && c2 == '|') { advance(); tokens.push_back(Token{TokenKind::KwOr, "||", 0, 0.0, tok_line, tok_col}); continue; }

        switch (c1) {
            case '(':
                paren_depth_++;
                tokens.push_back(Token{TokenKind::LParen, "(", 0, 0.0, tok_line, tok_col});
                break;
            case ')':
                if (paren_depth_ > 0) paren_depth_--;
                tokens.push_back(Token{TokenKind::RParen, ")", 0, 0.0, tok_line, tok_col});
                break;
            case '{':
                brace_depth_++;
                tokens.push_back(Token{TokenKind::LBrace, "{", 0, 0.0, tok_line, tok_col});
                break;
            case '}':
                if (brace_depth_ > 0) brace_depth_--;
                tokens.push_back(Token{TokenKind::RBrace, "}", 0, 0.0, tok_line, tok_col});
                if (brace_depth_ == 0 && paren_depth_ == 0) {
                    tokens.push_back(Token{TokenKind::Newline, "\\n", 0, 0.0, tok_line, tok_col});
                    at_line_start = true;
                }
                break;
            case '[':
                paren_depth_++;
                tokens.push_back(Token{TokenKind::LBracket, "[", 0, 0.0, tok_line, tok_col});
                break;
            case ']':
                if (paren_depth_ > 0) paren_depth_--;
                tokens.push_back(Token{TokenKind::RBracket, "]", 0, 0.0, tok_line, tok_col});
                break;
            case ',': tokens.push_back(Token{TokenKind::Comma, ",", 0, 0.0, tok_line, tok_col}); break;
            case ':': tokens.push_back(Token{TokenKind::Colon, ":", 0, 0.0, tok_line, tok_col}); break;
            case ';': tokens.push_back(Token{TokenKind::Semicolon, ";", 0, 0.0, tok_line, tok_col}); break;
            case '.': tokens.push_back(Token{TokenKind::Dot, ".", 0, 0.0, tok_line, tok_col}); break;
            case '?': tokens.push_back(Token{TokenKind::Question, "?", 0, 0.0, tok_line, tok_col}); break;
            case '+': tokens.push_back(Token{TokenKind::Plus, "+", 0, 0.0, tok_line, tok_col}); break;
            case '-': tokens.push_back(Token{TokenKind::Minus, "-", 0, 0.0, tok_line, tok_col}); break;
            case '*': tokens.push_back(Token{TokenKind::Star, "*", 0, 0.0, tok_line, tok_col}); break;
            case '/': tokens.push_back(Token{TokenKind::Slash, "/", 0, 0.0, tok_line, tok_col}); break;
            case '%': tokens.push_back(Token{TokenKind::Percent, "%", 0, 0.0, tok_line, tok_col}); break;
            case '=': tokens.push_back(Token{TokenKind::Assign, "=", 0, 0.0, tok_line, tok_col}); break;
            case '<': tokens.push_back(Token{TokenKind::Lt, "<", 0, 0.0, tok_line, tok_col}); break;
            case '>': tokens.push_back(Token{TokenKind::Gt, ">", 0, 0.0, tok_line, tok_col}); break;
            case '!': tokens.push_back(Token{TokenKind::KwNot, "!", 0, 0.0, tok_line, tok_col}); break;
            case '&': tokens.push_back(Token{TokenKind::Ampersand, "&", 0, 0.0, tok_line, tok_col}); break;
            case '|': tokens.push_back(Token{TokenKind::Pipe, "|", 0, 0.0, tok_line, tok_col}); break;
            case '^': tokens.push_back(Token{TokenKind::Caret, "^", 0, 0.0, tok_line, tok_col}); break;
            default: break;
        }
    }

    while (indent_stack_.size() > 1) {
        indent_stack_.pop_back();
        tokens.push_back(Token{TokenKind::Dedent, "<DEDENT>", 0, 0.0, line_, col_});
    }
    tokens.push_back(Token{TokenKind::Eof, "<EOF>", 0, 0.0, line_, col_});
    return tokens;
}

} // namespace flawn
