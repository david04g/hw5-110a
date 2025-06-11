# cse110A_parser.py
from enum import Enum
from cse110A_ast import *
from typing import List, Tuple, Optional
from scanner import Lexeme, Token, Scanner

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
        message = f"Symbol table error on line: {lineno}\nUndeclared ID: {ID}"
        super().__init__(message)

class NewLabelGenerator:
    def __init__(self) -> None:
        self.counter = 0
    def mk_new_label(self) -> str:
        lbl = f"label{self.counter}"
        self.counter += 1
        return lbl

class NewNameGenerator:
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []
    def mk_new_name(self) -> str:
        nm = f"_new_name{self.counter}"
        self.counter += 1
        self.new_names.append(nm)
        return nm

class VRAllocator:
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
        self.function_args = []

    def allocate_vrs(self, node: ASTNode) -> None:
        if is_leaf_node(node) and node.vr is None:
            node.vr = self.vra.mk_new_vr()
        elif is_binop_node(node):
            self.allocate_vrs(node.l_child)
            self.allocate_vrs(node.r_child)
            node.vr = self.vra.mk_new_vr()
        elif is_unop_node(node):
            self.allocate_vrs(node.child)
            node.vr = self.vra.mk_new_vr()

    def parse(self, s: str, uf: int) -> List[str]:
        self.scanner.input_string(s)
        self.to_match = self.scanner.token()
        self.uf = uf
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
        self.function_name = self.to_match.value
        self.eat(Token.ID)
        self.eat(Token.LPAR)
        self.function_args = self.parse_arg_list()
        self.eat(Token.RPAR)

    def parse_arg_list(self) -> List[Tuple[str, str]]:
        if self.get_token_id(self.to_match) == Token.RPAR:
            return []
        arg = self.parse_arg()
        if self.get_token_id(self.to_match) == Token.RPAR:
            return [arg]
        self.eat(Token.COMMA)
        return self.parse_arg_list() + [arg]

    def parse_arg(self) -> Tuple[str, str]:
        tok = self.get_token_id(self.to_match)
        if tok == Token.FLOAT:
            self.eat(Token.FLOAT)
            dt = Type.FLOAT
            ds = "float"
        elif tok == Token.INT:
            self.eat(Token.INT)
            dt = Type.INT
            ds = "int"
        else:
            raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
        nm = self.to_match.value
        self.eat(Token.ID)
        self.symbol_table.insert(nm, IDType.IO, dt)
        return (nm, ds)

    def parse_statement_list(self) -> List[str]:
        tok = self.get_token_id(self.to_match)
        if tok in [Token.INT, Token.FLOAT, Token.ID, Token.IF, Token.LBRACE, Token.FOR]:
            return self.parse_statement() + self.parse_statement_list()
        return []

    def parse_statement(self) -> List[str]:
        tok = self.get_token_id(self.to_match)
        if tok in [Token.INT, Token.FLOAT]:
            return self.parse_declaration_statement()
        if tok == Token.ID:
            return self.parse_assignment_statement()
        if tok == Token.IF:
            return self.parse_if_else_statement()
        if tok == Token.LBRACE:
            return self.parse_block_statement()
        if tok == Token.FOR:
            return self.parse_for_statement()
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.FOR, Token.IF, Token.LBRACE, Token.INT, Token.FLOAT, Token.ID])

    def parse_declaration_statement(self) -> List[str]:
        tok = self.get_token_id(self.to_match)
        if tok == Token.INT:
            self.eat(Token.INT)
            nm = self.to_match.value
            self.eat(Token.ID)
            self.eat(Token.SEMI)
            self.symbol_table.insert(nm, IDType.VAR, Type.INT, self.nng)
            return []
        if tok == Token.FLOAT:
            self.eat(Token.FLOAT)
            nm = self.to_match.value
            self.eat(Token.ID)
            self.eat(Token.SEMI)
            self.symbol_table.insert(nm, IDType.VAR, Type.FLOAT, self.nng)
            return []
        raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])

    def parse_assignment_statement_base(self, return_details: bool = False):
        nm = self.to_match.value
        data = self.symbol_table.lookup(nm)
        if data is None:
            raise SymbolTableException(self.scanner.get_lineno(), nm)
        self.eat(Token.ID)
        self.eat(Token.ASSIGN)
        ast = self.parse_expr()
        type_inference(ast)
        if data.data_type == Type.INT and ast.node_type == Type.FLOAT:
            conv = ASTFloatToIntNode(ast)
            conv.node_type = Type.INT
            ast = conv
        elif data.data_type == Type.FLOAT and ast.node_type == Type.INT:
            conv = ASTIntToFloatNode(ast)
            conv.node_type = Type.FLOAT
            ast = conv
        self.allocate_vrs(ast)
        pr = ast.linearize_code()
        if data.id_type == IDType.IO:
            inst = [f"{nm} = vr2int({ast.vr});"] if data.data_type == Type.INT else [f"{nm} = vr2float({ast.vr});"]
        else:
            inst = [f"{data.get_new_name()} = {ast.vr};"]
        seq = pr + inst
        return (seq, nm, ast) if return_details else seq

    def parse_assignment_statement(self) -> List[str]:
        seq = self.parse_assignment_statement_base()
        self.eat(Token.SEMI)
        return seq

    def parse_if_else_statement(self) -> List[str]:
        self.eat(Token.IF)
        self.eat(Token.LPAR)
        ast = self.parse_expr()
        type_inference(ast)
        self.allocate_vrs(ast)
        pr = ast.linearize_code()
        else_lbl = self.nlg.mk_new_label()
        end_lbl = self.nlg.mk_new_label()
        z = self.vra.mk_new_vr()
        ci = [f"{z} = int2vr(0);", f"beq({ast.vr}, {z}, {else_lbl});"]
        br = [f"branch({end_lbl});"]
        self.eat(Token.RPAR)
        ip = self.parse_statement()
        self.eat(Token.ELSE)
        ep = self.parse_statement()
        return pr + ci + ip + br + [f"{else_lbl}:"] + ep + [f"{end_lbl}:"]

    def parse_block_statement(self) -> List[str]:
        self.eat(Token.LBRACE)
        self.symbol_table.push_scope()
        seq = self.parse_statement_list()
        self.symbol_table.pop_scope()
        self.eat(Token.RBRACE)
        return seq

    def parse_for_statement(self) -> List[str]:
        self.eat(Token.FOR)
        self.eat(Token.LPAR)
        init_pr, init_nm, init_ast = self.parse_assignment_statement_base(True)
        self.eat(Token.SEMI)
        cond_ast = self.parse_expr()
        type_inference(cond_ast)
        self.allocate_vrs(cond_ast)
        cond_pr = cond_ast.linearize_code()
        self.eat(Token.SEMI)
        upd_pr, upd_nm, upd_ast = self.parse_assignment_statement_base(True)
        self.eat(Token.RPAR)
        body = self.parse_statement()
        data = self.symbol_table.lookup(init_nm)
        vstr = data.get_new_name() if data else init_nm
        uf = self.uf
        can = False
        if isinstance(init_ast, ASTNumNode) and init_ast.node_type == Type.INT:
            start = init_ast.value
            if isinstance(cond_ast, ASTLtNode) and cond_ast.l_child.get_name() == vstr and isinstance(cond_ast.r_child, ASTNumNode):
                bound = cond_ast.r_child.value
                if isinstance(upd_ast, ASTPlusNode) and upd_ast.l_child.get_name() == vstr and isinstance(upd_ast.r_child, ASTNumNode) and upd_ast.r_child.value == 1:
                    txt = "\n".join(body)
                    if all(x not in txt for x in ["beq", "branch", f"{vstr} ="]):
                        n = bound - start
                        if uf > 1 and n > 0 and n % uf == 0:
                            can = True
        ls = self.nlg.mk_new_label()
        le = self.nlg.mk_new_label()
        z = self.vra.mk_new_vr()
        ci = [f"{z} = int2vr(0);", f"beq({cond_ast.vr}, {z}, {le});"]
        br = [f"branch({ls});"]
        if can:
            un = []
            for j in range(uf):
                off = self.vra.mk_new_vr()
                un.append(f"{off} = int2vr({j});")
                ptr = self.vra.mk_new_vr()
                un.append(f"{ptr} = {vstr} + {off};")
                for ins in body:
                    un.append(ins.replace(vstr, ptr))
            step = self.vra.mk_new_vr()
            un.append(f"{step} = int2vr({uf});")
            un.append(f"{vstr} = {vstr} + {step};")
            return init_pr + [f"{ls}:"] + cond_pr + ci + un + br + [f"{le}:"]
        return init_pr + [f"{ls}:"] + cond_pr + ci + body + upd_pr + br + [f"{le}:"]

    def parse_expr(self) -> ASTNode:
        return self.parse_expr2(self.parse_comp())

    def parse_expr2(self, lhs: ASTNode) -> ASTNode:
        if self.get_token_id(self.to_match) == Token.EQ:
            self.eat(Token.EQ)
            rhs = self.parse_comp()
            return self.parse_expr2(ASTEqNode(lhs, rhs))
        return lhs

    def parse_comp(self) -> ASTNode:
        return self.parse_comp2(self.parse_factor())

    def parse_comp2(self, lhs: ASTNode) -> ASTNode:
        if self.get_token_id(self.to_match) == Token.LT:
            self.eat(Token.LT)
            rhs = self.parse_factor()
            return self.parse_comp2(ASTLtNode(lhs, rhs))
        return lhs

    def parse_factor(self) -> ASTNode:
        return self.parse_factor2(self.parse_term())

    def parse_factor2(self, lhs: ASTNode) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.PLUS:
            self.eat(Token.PLUS)
            rhs = self.parse_term()
            return self.parse_factor2(ASTPlusNode(lhs, rhs))
        if tok == Token.MINUS:
            self.eat(Token.MINUS)
            rhs = self.parse_term()
            return self.parse_factor2(ASTMinusNode(lhs, rhs))
        return lhs

    def parse_term(self) -> ASTNode:
        return self.parse_term2(self.parse_unit())

    def parse_term2(self, lhs: ASTNode) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.MUL:
            self.eat(Token.MUL)
            rhs = self.parse_unit()
            return self.parse_term2(ASTMultNode(lhs, rhs))
        if tok == Token.DIV:
            self.eat(Token.DIV)
            rhs = self.parse_unit()
            return self.parse_term2(ASTDivNode(lhs, rhs))
        return lhs

    def parse_unit(self) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.NUM:
            v = self.to_match.value
            self.eat(Token.NUM)
            return ASTNumNode(v)
        if tok == Token.ID:
            nm = self.to_match.value
            data = self.symbol_table.lookup(nm)
            if data is None:
                raise SymbolTableException(self.scanner.get_lineno(), nm)
            self.eat(Token.ID)
            return ASTIOIDNode(nm, data.data_type) if data.id_type == IDType.IO else ASTVarIDNode(data.get_new_name(), data.data_type)
        if tok == Token.LPAR:
            self.eat(Token.LPAR)
            node = self.parse_expr()
            self.eat(Token.RPAR)
            return node
        raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.NUM, Token.ID, Token.LPAR])

# Type inference helpers
def is_leaf_node(node: ASTNode) -> bool:
    return issubclass(type(node), ASTLeafNode)
def is_binop_node(node: ASTNode) -> bool:
    return issubclass(type(node), ASTBinOpNode)
def is_unop_node(node: ASTNode) -> bool:
    return issubclass(type(node), ASTUnOpNode)
def convert_children_type(node: ASTNode) -> None:
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
        elif isinstance(node, (ASTEqNode, ASTLtNode)):
            node.node_type = Type.INT
            convert_children_type(node)
    if is_unop_node(node):
        if isinstance(node, ASTIntToFloatNode):
            node.node_type = Type.FLOAT
        elif isinstance(node, ASTFloatToIntNode):
            node.node_type = Type.INT
    return node.node_type

