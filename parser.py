
import pdb
import class_ast as class_ast
from class_ast import *
from typing import Callable, List, Tuple, Optional
from scanner import Lexeme, Token, Scanner
from enum import Enum

class IDType(Enum):
    IO = 1
    VAR = 2

class SymbolTableData:
    def __init__(self, id_type: IDType, data_type: Type, new_name: str) -> None:
        self.id_type = id_type
        self.data_type = data_type
        self.new_name = new_name

    def get_id_type(self) -> IDType:
        return self.id_type

    def get_data_type(self) -> Type:
        return self.data_type

    def get_new_name(self) -> str:
        return self.new_name

class SymbolTableException(Exception):
    def __init__(self, lineno: int, ID: str) -> None:
        message = "Symbol table error on line: " + str(lineno) + "\nUndeclared ID: " + str(ID)
        super().__init__(message)

class NewLabelGenerator():
    def __init__(self) -> None:
        self.counter = 0

    def mk_new_label(self) -> str:
        new_label = "label" + str(self.counter)
        self.counter += 1
        return new_label

class NewNameGenerator():
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []

    def mk_new_name(self) -> str:
        new_name = "_new_name" + str(self.counter)
        self.counter += 1
        self.new_names.append(new_name)
        return new_name

class VRAllocator():
    def __init__(self) -> None:
        self.counter = 0

    def mk_new_vr(self) -> str:
        vr = "vr" + str(self.counter)
        self.counter += 1
        return vr

    def declare_variables(self) -> List[str]:
        ret = []
        for i in range(self.counter):
            ret.append("virtual_reg vr%d;" % i)
        return ret

class SymbolTable:
    def __init__(self) -> None:
        self.ht_stack = [dict()]

    def insert(self, ID: str, id_type: IDType, data_type: Type) -> None:
        info = SymbolTableData(id_type, data_type, ID)
        self.ht_stack[-1][ID] = info

    def lookup(self, ID: str) -> Optional:
        for ht in reversed(self.ht_stack):
            if ID in ht:
                return ht[ID]
        return None

    def push_scope(self) -> None:
        self.ht_stack.append(dict())

    def pop_scope(self) -> None:
        self.ht_stack.pop()

class ParserException(Exception):
    def __init__(self, lineno: int, lexeme: Lexeme, tokens: List[Token]) -> None:
        message = "Parser error on line: " + str(lineno) + "\nExpected one of: " + str(tokens) + "\nGot: " + str(lexeme)
        super().__init__(message)

class Parser:
    def __init__(self, scanner: Scanner) -> None:
        self.scanner = scanner
        self.symbol_table = SymbolTable()
        self.vra = VRAllocator()
        self.nlg = NewLabelGenerator()
        self.nng = NewNameGenerator()
        self.function_name = None
        self.function_args = []
        self.unroll_factor = 1  # default, override externally if needed

    def parse(self, s: str) -> List[str]:
        self.scanner.input_string(s)
        self.to_match = self.scanner.token()
        p = self.parse_function()
        self.eat(None)
        return p

    def get_token_id(self, l: Lexeme) -> Token:
        if l is None:
            return None
        return l.token

    def eat(self, check: Token) -> None:
        token_id = self.get_token_id(self.to_match)
        if token_id != check:
            raise ParserException(self.scanner.get_lineno(), self.to_match, [check])
        self.to_match = self.scanner.token()

    def parse_function(self) -> List[str]:
        self.parse_function_header()
        self.eat(Token.LBRACE)
        p = self.parse_statement_list()
        self.eat(Token.RBRACE)
        return p

    def parse_function_header(self) -> None:
        self.eat(Token.VOID)
        function_name = self.to_match.value
        self.eat(Token.ID)
        self.eat(Token.LPAR)
        self.function_name = function_name
        args = self.parse_arg_list()
        self.function_args = args
        self.eat(Token.RPAR)

    def parse_arg_list(self) -> List[Tuple[str, str]]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.RPAR:
            return []
        arg = self.parse_arg()
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.RPAR:
            return [arg]
        self.eat(Token.COMMA)
        arg_l = self.parse_arg_list()
        return arg_l + [arg]

    def parse_arg(self) -> Tuple[str, str]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.FLOAT:
            self.eat(Token.FLOAT)
            data_type = Type.FLOAT
            data_type_str = "float"
        elif token_id == Token.INT:
            self.eat(Token.INT)
            data_type = Type.INT
            data_type_str = "int"
        else:
            raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
        id_name = self.to_match.value
        self.eat(Token.ID)
        self.symbol_table.insert(id_name, IDType.IO, data_type)
        return (id_name, data_type_str)

def parse_for_statement(self) -> List[str]:
    self.eat(Token.FOR)
    self.eat(Token.LPAR)

    # Parse init assignment
    init_code = self.parse_assignment_statement()
    init_assign = init_code[0] if init_code else ""
    self.eat(Token.SEMI)

    # Parse condition
    cond_ast = self.parse_expr()
    self.eat(Token.SEMI)

    # Parse update assignment
    update_code = self.parse_assignment_statement_base()
    update_assign = update_code[0] if update_code else ""
    self.eat(Token.RPAR)

    # Parse body
    self.symbol_table.push_scope()
    body_code = self.parse_statement()
    self.symbol_table.pop_scope()

    # Attempt unrolling if pattern matches
    try:
        if (cond_ast.op == '<' and isinstance(cond_ast.left, ASTIDNode) and isinstance(cond_ast.right, ASTNUMNode)):
            loop_var = cond_ast.left.name
            limit = cond_ast.right.value

            if isinstance(init_assign, str) and f"{loop_var}=" in init_assign:
                start_val = int(init_assign.split('=')[1].strip())
                if isinstance(update_assign, str) and update_assign.strip() == f"{loop_var}={loop_var}+1":
                    unroll_factor = getattr(self, 'unroll_factor', 1)
                    trip_count = limit - start_val
                    if trip_count % unroll_factor == 0:
                        result = []
                        result.append(init_assign)
                        for i in range(start_val, limit, unroll_factor):
                            for j in range(unroll_factor):
                                iter_val = i + j
                                for stmt in body_code:
                                    result.append(stmt.replace(loop_var, str(iter_val)))
                        return result
    except:
        pass

    # Fallback if not unrolled
    label_start = self.nlg.mk_new_label()
    label_end = self.nlg.mk_new_label()
    cond_code = ["ifFalse " + cond_ast.to_code() + " goto " + label_end]
    return init_code + [label_start + ":"] + cond_code + body_code + update_code + ["goto " + label_start, label_end + ":"]
