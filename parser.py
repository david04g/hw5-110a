import pdb
import class_ast as class_ast
from class_ast import *
from typing import Callable, List, Tuple, Optional
from scanner import Lexeme, Token, Scanner
from enum import Enum

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
        self.id_type = id_type      # IO or VAR
        self.data_type = data_type  # INT or FLOAT
        self.new_name = new_name    # renaming for scoping

    def get_id_type(self) -> IDType:
        return self.id_type

    def get_data_type(self) -> Type:
        return self.data_type

    def get_new_name(self) -> str:
        return self.new_name

# Symbol Table exception, requires a line number and ID
class SymbolTableException(Exception):
    def __init__(self, lineno: int, ID: str) -> None:
        message = "Symbol table error on line: " + str(lineno) + \
                  "\nUndeclared ID: " + str(ID)
        super().__init__(message)

# Generates new labels
class NewLabelGenerator():
    def __init__(self) -> None:
        self.counter = 0

    def mk_new_label(self) -> str:
        new_label = "label" + str(self.counter)
        self.counter += 1
        return new_label

# Generates new names for program variables
class NewNameGenerator():
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []

    def mk_new_name(self) -> str:
        new_name = "_new_name" + str(self.counter)
        self.counter += 1
        self.new_names.append(new_name)
        return new_name

# Allocates virtual registers
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
            ret.append(f"virtual_reg vr{i};")
        return ret

# Symbol table class
class SymbolTable:
    def __init__(self) -> None:
        self.ht_stack = [dict()]

    def insert(self, ID: str, id_type: IDType, data_type: Type) -> None:
        info = SymbolTableData(id_type, data_type, ID)
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

# Parser Exception
class ParserException(Exception):
    def __init__(self, lineno: int, lexeme: Lexeme, tokens: List[Token]) -> None:
        message = ("Parser error on line: " + str(lineno) +
                   "\nExpected one of: " + str(tokens) +
                   "\nGot: " + str(lexeme))
        super().__init__(message)

# Parser class
class Parser:

    def __init__(self, scanner: Scanner) -> None:
        self.scanner = scanner
        self.symbol_table = SymbolTable()
        self.vra = VRAllocator()
        self.nlg = NewLabelGenerator()
        self.nng = NewNameGenerator()
        self.function_name = None
        self.function_args: List[Tuple[str,str]] = []
        self.unroll_factor = 1

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
            raise ParserException(self.scanner.get_lineno(),
                                  self.to_match,
                                  [check])
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
        args = self.parse_arg_list() or []
        self.function_args = args
        self.eat(Token.RPAR)

    def parse_arg_list(self) -> List[Tuple[str,str]]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.RPAR:
            return []
        arg = self.parse_arg()
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.RPAR:
            return [arg]
        self.eat(Token.COMMA)
        return self.parse_arg_list() + [arg]

    def parse_arg(self) -> Tuple[str,str]:
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
            raise ParserException(self.scanner.get_lineno(),
                                  self.to_match,
                                  [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
        id_name = self.to_match.value
        self.eat(Token.ID)
        self.symbol_table.insert(id_name, IDType.IO, data_type)
        return (id_name, data_type_str)

    def parse_statement_list(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT, Token.FLOAT, Token.ID, Token.IF, Token.LBRACE, Token.FOR]:
            stm = self.parse_statement() or []
            return stm + self.parse_statement_list()
        return []

    def parse_statement(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT, Token.FLOAT]:
            return self.parse_declaration_statement() or []
        if token_id == Token.ID:
            return self.parse_assignment_statement() or []
        if token_id == Token.IF:
            return self.parse_if_else_statement() or []
        if token_id == Token.LBRACE:
            return self.parse_block_statement() or []
        if token_id == Token.FOR:
            return self.parse_for_statement() or []
        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,
                              [Token.FOR, Token.IF, Token.LBRACE,
                               Token.INT, Token.FLOAT, Token.ID])

    def parse_declaration_statement(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.INT:
            self.eat(Token.INT)
            id_name = self.to_match.value
            self.eat(Token.ID)
            self.symbol_table.insert(id_name, IDType.VAR, Type.INT)
            self.eat(Token.SEMI)
            return []
        if token_id == Token.FLOAT:
            self.eat(Token.FLOAT)
            id_name = self.to_match.value
            self.eat(Token.ID)
            self.symbol_table.insert(id_name, IDType.VAR, Type.FLOAT)
            self.eat(Token.SEMI)
            return []
        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,
                              [Token.INT, Token.FLOAT])

    def parse_assignment_statement(self) -> List[str]:
        base = self.parse_assignment_statement_base()
        self.eat(Token.SEMI)
        return base or []

    def parse_assignment_statement_base(self) -> List[str]:
        id_name = self.to_match.value
        id_data = self.symbol_table.lookup(id_name)
        if id_data is None:
            raise SymbolTableException(self.scanner.get_lineno(), id_name)
        self.eat(Token.ID)
        self.eat(Token.ASSIGN)
        self.parse_expr()
        return []

    def parse_if_else_statement(self) -> List[str]:
        self.eat(Token.IF)
        self.eat(Token.LPAR)
        self.parse_expr()
        self.eat(Token.RPAR)
        self.parse_statement()
        self.eat(Token.ELSE)
        self.parse_statement()
        return []

    def parse_block_statement(self) -> List[str]:
        self.eat(Token.LBRACE)
        self.symbol_table.push_scope()
        body = self.parse_statement_list()
        self.symbol_table.pop_scope()
        self.eat(Token.RBRACE)
        return []

    def parse_for_statement(self) -> List[str]:
        self.eat(Token.FOR)
        self.eat(Token.LPAR)

        # initializer
        init_code = self.parse_assignment_statement()
        self.eat(Token.SEMI)

        # condition
        cond_ast = self.parse_expr()
        self.eat(Token.SEMI)

        # update
        update_code = self.parse_assignment_statement_base()
        self.eat(Token.RPAR)

        # body
        self.symbol_table.push_scope()
        body_code = self.parse_statement() or []
        self.symbol_table.pop_scope()

        # loop unrolling
        try:
            if (isinstance(cond_ast, ASTBinOpNode)
                and cond_ast.op == '<'
                and isinstance(cond_ast.left, ASTIDNode)
                and isinstance(cond_ast.right, ASTNUMNode)):

                var = cond_ast.left.name
                limit = cond_ast.right.value

                init_str = init_code[0].strip() if len(init_code)==1 else ""
                upd_str  = update_code[0].strip() if len(update_code)==1 else ""

                if init_str.startswith(f"{var}=") and upd_str == f"{var}={var}+1":
                    start = int(init_str.split('=')[1])
                    uf = self.unroll_factor
                    count = limit - start
                    if count % uf == 0:
                        out = []
                        for v in range(start, limit, uf):
                            for j in range(uf):
                                iv = v + j
                                for ln in body_code:
                                    out.append(ln.replace(var, str(iv)))
                        return out
        except:
            pass

        # fallback
        L0 = self.nlg.mk_new_label()
        L1 = self.nlg.mk_new_label()
        cond_line = f"ifFalse {cond_ast.to_code()} goto {L1}"
        return init_code + [f"{L0}:"] + [cond_line] + body_code + update_code + [f"goto {L0}", f"{L1}:"]
# you need to build and return an AST
    def parse_expr(self) -> ASTNode:        
        self.parse_comp()
        self.parse_expr2()
        return

    # you need to build and return an AST
    def parse_expr2(self) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.EQ]:
            self.eat(Token.EQ)
            self.parse_comp()
            self.parse_expr2()
            return
        if token_id in [Token.SEMI, Token.RPAR]:
            return 
        
        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.EQ, Token.SEMI, Token.RPAR])
    
    # you need to build and return an AST
    def parse_comp(self) -> ASTNode:
        self.parse_factor()
        self.parse_comp2()
        return

    # you need to build and return an AST
    def parse_comp2(self) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.LT]:
            self.eat(Token.LT)
            self.parse_factor()
            self.parse_comp2()
            return
        if token_id in [Token.SEMI, Token.RPAR, Token.EQ]:
            return 
        
        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.EQ, Token.SEMI, Token.RPAR, Token.LT])

    # you need to build and return an AST
    def parse_factor(self) -> ASTNode:
        self.parse_term()
        self.parse_factor2()
        return

    # you need to build and return an AST
    def parse_factor2(self) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.PLUS]:
            self.eat(Token.PLUS)
            self.parse_term()            
            self.parse_factor2()
            return
        if token_id in [Token.MINUS]:
            self.eat(Token.MINUS)
            self.parse_term()
            self.parse_factor2()
            return
        if token_id in [Token.EQ, Token.SEMI, Token.RPAR, Token.LT]:
            return

        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.EQ, Token.SEMI, Token.RPAR, Token.LT, Token.PLUS, Token.MINUS])
    
    # you need to build and return an AST
    def parse_term(self) -> ASTNode:
        self.parse_unit()
        self.parse_term2()
        return

    # you need to build and return an AST
    def parse_term2(self) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.DIV]:
            self.eat(Token.DIV)
            self.parse_unit()
            self.parse_term2()
            return 
        if token_id in [Token.MUL]:
            self.eat(Token.MUL)
            self.parse_unit()
            self.parse_term2()
            return 
        if token_id in [Token.EQ, Token.SEMI, Token.RPAR, Token.LT, Token.PLUS, Token.MINUS]:
            return 

        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.EQ, Token.SEMI, Token.RPAR, Token.LT, Token.PLUS, Token.MINUS, Token.MUL, Token.DIV])

    # you need to build and return an AST
    def parse_unit(self) -> ASTNode:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.NUM]:
            self.eat(Token.NUM)            
            return
        if token_id in [Token.ID]:
            id_name = self.to_match.value
            id_data = self.symbol_table.lookup(id_name)
            if id_data == None:
                raise SymbolTableException(self.scanner.get_lineno(), id_name)
            self.eat(Token.ID)
            return
        if token_id in [Token.LPAR]:
            self.eat(Token.LPAR)
            self.parse_expr()
            self.eat(Token.RPAR)
            return
            
        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.NUM, Token.ID, Token.LPAR])    

# Type inference start

# I suggest making functions like this to check
# what class a node belongs to.
def is_leaf_node(node) -> bool:
    return issubclass(type(node), ASTLeafNode)

# Type inference top level
def type_inference(node) -> Type:
    
    if is_leaf_node(node):
        return node.node_type
    
    # next check if it is a unary op, then a bin op.
    # remember that comparison operators (eq and lt)
    # are handled a bit differently
