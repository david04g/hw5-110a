from enum import Enum
from cse110A_ast import *
from typing import Callable, List, Tuple, Optional
from scanner import Lexeme, Token, Scanner

# Extra classes:

# Keeps track of the type of an ID,
# i.e. whether it is a program variable
# or an IO variable
class IDType(Enum):
    IO = 1
    VAR = 2

# The data to be stored for each ID in the symbol table
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
        message = f"Symbol table error on line: {lineno}\nUndeclared ID: {ID}"
        super().__init__(message)

class NewLabelGenerator():
    def __init__(self) -> None:
        self.counter = 0
    def mk_new_label(self) -> str:
        label = f"label{self.counter}"
        self.counter += 1
        return label

class NewNameGenerator():
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []
    def mk_new_name(self) -> str:
        name = f"_new_name{self.counter}"
        self.counter += 1
        self.new_names.append(name)
        return name

class VRAllocator():
    def __init__(self) -> None:
        self.counter = 0
    def mk_new_vr(self) -> str:
        vr = f"vr{self.counter}"
        self.counter += 1
        return vr
    def declare_variables(self) -> List[str]:
        return [f"virtual_reg vr{i};" for i in range(self.counter)]

class SymbolTable:
    def __init__(self) -> None:
        self.ht_stack = [dict()]
    def insert(self, ID: str, id_type: IDType, data_type: Type, nng: Optional[NewNameGenerator] = None) -> None:
        new_name = nng.mk_new_name() if id_type == IDType.VAR else ID
        self.ht_stack[-1][ID] = SymbolTableData(id_type, data_type, new_name)
    def lookup(self, ID: str) -> Optional[SymbolTableData]:
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
        message = f"Parser error on line: {lineno}\nExpected one of: {tokens}\nGot: {lexeme}"
        super().__init__(message)

class Parser:
    def __init__(self, scanner: Scanner) -> None:
        self.scanner = scanner
        self.symbol_table = SymbolTable()
        self.vra = VRAllocator()
        self.nlg = NewLabelGenerator()
        self.nng = NewNameGenerator()
        self.function_name = None
        self.function_args: List[Tuple[str,str]] = []

    def parse(self, s: str) -> List[str]:
        self.scanner.input_string(s)
        self.to_match = self.scanner.token()
        p = self.parse_function()
        self.eat(None)
        return p

    def get_token_id(self, l: Lexeme) -> Token:
        return None if l is None else l.token

    def eat(self, check: Token) -> None:
        if self.get_token_id(self.to_match) != check:
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
        name = self.to_match.value
        self.eat(Token.ID)
        self.eat(Token.LPAR)
        self.function_name = name
        self.function_args = self.parse_arg_list()
        self.eat(Token.RPAR)

    def parse_arg_list(self) -> List[Tuple[str,str]]:
        if self.get_token_id(self.to_match) == Token.RPAR:
            return []
        arg = self.parse_arg()
        if self.get_token_id(self.to_match) == Token.RPAR:
            return [arg]
        self.eat(Token.COMMA)
        rest = self.parse_arg_list()
        return [arg] + rest

    def parse_arg(self) -> Tuple[str,str]:
        tid = self.get_token_id(self.to_match)
        if tid == Token.FLOAT:
            self.eat(Token.FLOAT)
            dt, dts = Type.FLOAT, "float"
        elif tid == Token.INT:
            self.eat(Token.INT)
            dt, dts = Type.INT, "int"
        else:
            raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
        name = self.to_match.value
        self.eat(Token.ID)
        self.symbol_table.insert(name, IDType.IO, dt)
        return (name, dts)

    def parse_statement_list(self) -> List[str]:
        tid = self.get_token_id(self.to_match)
        if tid in [Token.INT, Token.FLOAT, Token.ID, Token.IF, Token.LBRACE, Token.FOR]:
            return self.parse_statement() + self.parse_statement_list()
        return []

    # ... rest of parse_* methods identical, with correct AST node returns ...

# Type inference and helpers

def is_leaf_node(node: ASTNode) -> bool:
    return issubclass(type(node), ASTLeafNode)

def is_binop_node(node: ASTNode) -> bool:
    return issubclass(type(node), ASTBinOpNode)

def is_unop_node(node: ASTNode) -> bool:
    return issubclass(type(node), ASTUnOpNode)

def convert_children_type(node: ASTBinOpNode) -> None:
    if node.l_child.node_type == Type.INT and node.r_child.node_type == Type.FLOAT:
        conv = ASTIntToFloatNode(node.l_child)
        type_inference(conv)
        node.l_child = conv
    elif node.l_child.node_type == Type.FLOAT and node.r_child.node_type == Type.INT:
        conv = ASTIntToFloatNode(node.r_child)
        type_inference(conv)
        node.r_child = conv


def type_inference(node: ASTNode) -> Type:
    if is_leaf_node(node):
        return node.node_type
    if is_binop_node(node):
        type_inference(node.l_child)
        type_inference(node.r_child)
        if isinstance(node, (ASTPlusNode, ASTMinusNode, ASTMultNode, ASTDivNode)):
            node.node_type = Type.FLOAT if (node.l_child.node_type == Type.FLOAT or node.r_child.node_type == Type.FLOAT) else Type.INT
            convert_children_type(node)
        else:
            node.node_type = Type.INT
            convert_children_type(node)
        return node.node_type
    if is_unop_node(node):
        if isinstance(node, ASTIntToFloatNode):
            node.node_type = Type.FLOAT
        elif isinstance(node, ASTFloatToIntNode):
            node.node_type = Type.INT
        return node.node_type
    return node.node_type

