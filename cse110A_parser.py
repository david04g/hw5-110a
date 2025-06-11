# cse110a_parser.py
from cse110A_ast import *
from typing import Callable, List, Tuple, Optional
from scanner import Lexeme, Token, Scanner
from enum import Enum

# Extra classes:
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

class NewLabelGenerator():
    def __init__(self) -> None:
        self.counter = 0
    def mk_new_label(self) -> str:
        lbl = f"label{self.counter}"
        self.counter += 1
        return lbl

class NewNameGenerator():
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []
    def mk_new_name(self) -> str:
        nm = f"_new_name{self.counter}"
        self.counter += 1
        self.new_names.append(nm)
        return nm

class VRAllocator():
    def __init__(self) -> None:
        self.counter = 0
    def mk_new_vr(self) -> str:
        vr = f"vr{self.counter}"
        self.counter += 1
        return vr
    def declare_variables(self) -> List[str]:
        ret = []
        for i in range(self.counter):
            ret.append(f"virtual_reg vr{i};")
        return ret

class SymbolTable:
    def __init__(self) -> None:
        self.ht_stack = [dict()]
    def insert(self, ID: str, id_type: IDType, data_type: Type, nng: Optional[NewNameGenerator] = None) -> None:
        if id_type == IDType.VAR:
            new_name = nng.mk_new_name()
        else:
            new_name = ID
        info = SymbolTableData(id_type, data_type, new_name)
        self.ht_stack[-1][ID] = info
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
        message = (f"Parser error on line: {lineno}\nExpected one of: {tokens}\nGot: {lexeme}")
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
        self.function_name = self.to_match.value
        self.eat(Token.ID)
        self.eat(Token.LPAR)
        args = self.parse_arg_list()
        self.function_args = args
        self.eat(Token.RPAR)

    def parse_arg_list(self) -> List[Tuple[str,str]]:
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
            data_type = Type.FLOAT
            data_str = "float"
        elif tok == Token.INT:
            self.eat(Token.INT)
            data_type = Type.INT
            data_str = "int"
        else:
            raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
        id_name = self.to_match.value
        self.eat(Token.ID)
        self.symbol_table.insert(id_name, IDType.IO, data_type)
        return (id_name, data_str)

    def parse_statement_list(self) -> List[str]:
        tok = self.get_token_id(self.to_match)
        if tok in [Token.INT, Token.FLOAT, Token.ID, Token.IF, Token.LBRACE, Token.FOR]:
            a = self.parse_statement()
            b = self.parse_statement_list()
            return a + b
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
            id_name = self.to_match.value
            self.eat(Token.ID)
            self.eat(Token.SEMI)
            self.symbol_table.insert(id_name, IDType.VAR, Type.INT, self.nng)
            return []
        if tok == Token.FLOAT:
            self.eat(Token.FLOAT)
            id_name = self.to_match.value
            self.eat(Token.ID)
            self.eat(Token.SEMI)
            self.symbol_table.insert(id_name, IDType.VAR, Type.FLOAT, self.nng)
            return []
        raise ParserException(self.scanner.get_lineno(), self.to_match, [Token.INT, Token.FLOAT])

    def parse_assignment_statement_base(self, return_details: bool=False):
        id_name = self.to_match.value
        id_data = self.symbol_table.lookup(id_name)
        if id_data is None:
            raise SymbolTableException(self.scanner.get_lineno(), id_name)
        self.eat(Token.ID)
        self.eat(Token.ASSIGN)
        expr_ast = self.parse_expr()
        type_inference(expr_ast)
        if id_data.data_type == Type.INT and expr_ast.node_type == Type.FLOAT:
            new_root = ASTFloatToIntNode(expr_ast)
            new_root.node_type = Type.INT
            expr_ast = new_root
        elif id_data.data_type == Type.FLOAT and expr_ast.node_type == Type.INT:
            new_root = ASTIntToFloatNode(expr_ast)
            new_root.node_type = Type.FLOAT
            expr_ast = new_root
        self.allocate_vrs(expr_ast)
        program = expr_ast.linearize_code()
        if id_data.id_type == IDType.IO:
            if id_data.data_type == Type.INT:
                inst = [f"{id_name} = vr2int({expr_ast.vr});"]
            else:
                inst = [f"{id_name} = vr2float({expr_ast.vr});"]
        else:
            inst = [f"{id_data.new_name} = {expr_ast.vr};"]
        inst_list = program + inst
        if return_details:
            return inst_list, id_name, expr_ast
        return inst_list

    def parse_assignment_statement(self) -> List[str]:
        i = self.parse_assignment_statement_base()
        self.eat(Token.SEMI)
        return i

    def parse_if_else_statement(self) -> List[str]:
        self.eat(Token.IF)
        self.eat(Token.LPAR)
        expr_ast = self.parse_expr()
        type_inference(expr_ast)
        self.allocate_vrs(expr_ast)
        expr_prog = expr_ast.linearize_code()
        else_lbl = self.nlg.mk_new_label()
        end_lbl = self.nlg.mk_new_label()
        zero_vr = self.vra.mk_new_vr()
        cmp_ins = [f"{zero_vr} = int2vr(0);", f"beq({expr_ast.vr}, {zero_vr}, {else_lbl});"]
        br_ins  = [f"branch({end_lbl});"]
        self.eat(Token.RPAR)
        if_prog = self.parse_statement()
        self.eat(Token.ELSE)
        else_prog = self.parse_statement()
        return expr_prog + cmp_ins + if_prog + br_ins + [f"{else_lbl}:"] + else_prog + [f"{end_lbl}:"]

    def parse_block_statement(self) -> List[str]:
        self.eat(Token.LBRACE)
        self.symbol_table.push_scope()
        ret = self.parse_statement_list()
        self.symbol_table.pop_scope()
        self.eat(Token.RBRACE)
        return ret

    def parse_for_statement(self) -> List[str]:
        self.eat(Token.FOR)
        self.eat(Token.LPAR)
        init_prog, init_var, init_ast = self.parse_assignment_statement_base(return_details=True)
        self.eat(Token.SEMI)
        cond_ast = self.parse_expr()
        type_inference(cond_ast)
        self.allocate_vrs(cond_ast)
        cond_prog = cond_ast.linearize_code()
        self.eat(Token.SEMI)
        update_prog, update_var, update_ast = self.parse_assignment_statement_base(return_details=True)
        self.eat(Token.RPAR)
        body_prog = self.parse_statement()
        uf = self.uf
        can_unroll = False
        if isinstance(init_ast, ASTNumNode) and init_ast.node_type == Type.INT:
            start = init_ast.value
            if (isinstance(cond_ast, ASTLtNode)
                and isinstance(cond_ast.l_child, ASTVarIDNode)
                and cond_ast.l_child.get_name() == init_var
                and isinstance(cond_ast.r_child, ASTNumNode)
                and cond_ast.r_child.node_type == Type.INT):
                bound = cond_ast.r_child.value
                if (isinstance(update_ast, ASTPlusNode)
                    and isinstance(update_ast.l_child, ASTVarIDNode)
                    and update_ast.l_child.get_name() == init_var
                    and isinstance(update_ast.r_child, ASTNumNode)
                    and update_ast.r_child.value == 1):
                    body_text = "\n".join(body_prog)
                    if "beq" not in body_text and "branch" not in body_text and f"{init_var} =" not in body_text:
                        n_iters = bound - start
                        if uf > 1 and n_iters > 0 and n_iters % uf == 0:
                            can_unroll = True
        lg = self.nlg; vr = self.vra
        loop_start = lg.mk_new_label()
        loop_end   = lg.mk_new_label()
        zero_vr    = vr.mk_new_vr()
        cmp_ins    = [f"{zero_vr} = int2vr(0);", f"beq({cond_ast.vr}, {zero_vr}, {loop_end});"]
        br_ins     = [f"branch({loop_start});"]
        if can_unroll:
            unrolled = []
            for j in range(uf):
                off = vr.mk_new_vr()
                unrolled.append(f"{off} = int2vr({j});")
                ptr = vr.mk_new_vr()
                unrolled.append(f"{ptr} = {init_var} + {off};")
                for inst in body_prog:
                    unrolled.append(inst.replace(init_var, ptr))
            step = vr.mk_new_vr()
            unrolled.append(f"{step} = int2vr({uf});")
            unrolled.append(f"{init_var} = {init_var} + {step};")
            return (
                init_prog +
                [f"{loop_start}:"] +
                cond_prog +
                cmp_ins +
                unrolled +
                br_ins +
                [f"{loop_end}:"]
            )
        return (
            init_prog +
            [f"{loop_start}:"] +
            cond_prog +
            cmp_ins +
            body_prog +
            update_prog +
            br_ins +
            [f"{loop_end}:"]
        )

    def parse_expr(self) -> ASTNode:
        n = self.parse_comp()
        return self.parse_expr2(n)
    def parse_expr2(self, lhs: ASTNode) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.EQ:
            self.eat(Token.EQ)
            r = self.parse_comp()
            return self.parse_expr2(ASTEqNode(lhs, r))
        return lhs
    def parse_comp(self) -> ASTNode:
        n = self.parse_factor()
        return self.parse_comp2(n)
    def parse_comp2(self, lhs: ASTNode) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.LT:
            self.eat(Token.LT); r = self.parse_factor()
            return self.parse_comp2(ASTLtNode(lhs, r))
        return lhs
    def parse_factor(self, node=None) -> ASTNode:
        n = self.parse_term() if node is None else node
        return self.parse_factor2(n)
    def parse_factor2(self, lhs: ASTNode) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.PLUS:
            self.eat(Token.PLUS); r = self.parse_term()
            return self.parse_factor2(ASTPlusNode(lhs, r))
        if tok == Token.MINUS:
            self.eat(Token.MINUS); r = self.parse_term()
            return self.parse_factor2(ASTMinusNode(lhs, r))
        return lhs
    def parse_term(self) -> ASTNode:
        n = self.parse_unit()
        return self.parse_term2(n)
    def parse_term2(self, lhs: ASTNode) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.MUL:
            self.eat(Token.MUL); r = self.parse_unit()
            return self.parse_term2(ASTMultNode(lhs, r))
        if tok == Token.DIV:
            self.eat(Token.DIV); r = self.parse_unit()
            return self.parse_term2(ASTDivNode(lhs, r))
        return lhs
    def parse_unit(self) -> ASTNode:
        tok = self.get_token_id(self.to_match)
        if tok == Token.NUM:
            v = self.to_match.value
            node = ASTNumNode(v)
            self.eat(Token.NUM)
            return node
        if tok == Token.ID:
            name = self.to_match.value
            data = self.symbol_table.lookup(name)
            if data is None:
                raise SymbolTableException(self.scanner.get_lineno(), name)
            self.eat(Token.ID)
            if data.id_type == IDType.IO:
                return ASTIOIDNode(name, data.data_type)
            return ASTVarIDNode(data.new_name, data.data_type)
        if tok == Token.LPAR:
            self.eat(Token.LPAR)
            n = self.parse_expr()
            self.eat(Token.RPAR)
            return n
        raise ParserException(self.scanner.get_lineno(), self.to_match,
                              [Token.NUM, Token.ID, Token.LPAR])

