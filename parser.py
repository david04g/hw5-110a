# change types to type ~ change IDtypes to IDtype
import pdb
import class_ast as class_ast
from class_ast import *
from typing import Callable,List,Tuple,Optional
from scanner import Lexeme,Token,Scanner

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
        self.id_type = id_type      # if the variable is input/output
                                    # or variable
                                    
        self.data_type = data_type  # if the variable is an int or
                                    # float
                                    
        self.new_name = new_name    # a new name to resolve collisions
                                    # in scoping

    # Getters for each of the elements
    def get_id_type(self) -> IDType:
        return self.id_type

    def get_data_type(self) -> Type:
        return self.data_type

    def get_new_name(self) -> str:
        return self.new_name

# Symbol Table exception, requires a line number and ID
class SymbolTableException(Exception):
    def __init__(self, lineno: int, ID: str) -> None:
        message = "Symbol table error on line: " + str(lineno) + "\nUndeclared ID: " + str(ID)
        super().__init__(message)

# Generates a new label when needed
class NewLabelGenerator():
    def __init__(self) -> None:
        self.counter = 0
        
    def mk_new_label(self) -> str:
        new_label = "label" + str(self.counter)
        self.counter += 1
        return new_label

# Generates a new name (e.g. for program variables)
# when needed
class NewNameGenerator():
    def __init__(self) -> None:
        self.counter = 0
        self.new_names = []

    # You may want to make a better renaming scheme
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

    # get variable declarations (needed for the C++ wrapper)
    def declare_variables(self) -> List[str]:
        ret = []
        for i in range(self.counter):
            ret.append("virtual_reg vr%d;" % i)

        return ret

# Symbol table class
class SymbolTable:
    def __init__(self) -> None:
        # stack of hashtables
        self.ht_stack = [dict()]

    def insert(self, ID: str, id_type: IDType, data_type: Type) -> None:
        
        # Create the data to store for the ID
        
        # HOMEWORK: make sure this is storing the
        # right information! You may need to use
        # the new name generator to make new names
        # for some ids
        info = SymbolTableData(id_type, data_type, ID)
        self.ht_stack[-1][ID] = info        

    # Lookup the symbol. If it is there, return the
    # info, otherwise return Noney
    def lookup(self, ID: str) -> Optional:
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
    
    # Pass a line number, current lexeme, and what tokens are expected
    def __init__(self, lineno: int, lexeme: Lexeme, tokens: List[Token]) -> None:
        message = "Parser error on line: " + str(lineno) + "\nExpected one of: " + str(tokens) + "\nGot: " + str(lexeme)
        super().__init__(message)

# Parser class
class Parser:

    # Creating the parser requires a scanner
    def __init__(self, scanner: Scanner) -> None:
        
        self.scanner = scanner

        # Create a symbol table
        self.symbol_table = SymbolTable()

        # objects to create virtual registers,
        # labels, and new names
        self.vra = VRAllocator()
        self.nlg = NewLabelGenerator()
        self.nng = NewNameGenerator()

        # needed to create the C++ wrapper
        # You do not need to modify these for the
        # homework
        self.function_name = None
        self.function_args = []

    # HOMEWORK: top level function:
    # This needs to return a list of 3 address instructions
    def parse(self, s: str) -> List[str]:

        # Set the scanner and get the first token
        self.scanner.input_string(s)
        self.to_match = self.scanner.token()

        # start parsing. In your solution, p must contain a list of
        # three address instructions
        p = self.parse_function()
        self.eat(None)
        
        return p

    # Helper fuction: get the token ID
    def get_token_id(self, l: Lexeme) ->Token:
        if l is None:
            return None
        return l.token

    # Helper fuction: eat a token ID and advance
    # to the next token
    def eat(self, check: Token) -> None:
        token_id = self.get_token_id(self.to_match)
        if token_id != check:
            raise ParserException(self.scanner.get_lineno(),
                                  self.to_match,
                                  [check])      
        self.to_match = self.scanner.token()

    # The top level parse_function
    def parse_function(self) -> List[str]:

        # I am parsing the function header for you
        # You do not need to do anything with this.
        self.parse_function_header()    
        self.eat(Token.LBRACE)

        # your solution should have p containing a list
        # of three address instructions
        p = self.parse_statement_list()        
        self.eat(Token.RBRACE)
        return p

    # You do not need to modify this for your homework
    # but you can look :) 
    def parse_function_header(self) -> None:
        self.eat(Token.VOID)
        function_name = self.to_match.value
        self.eat(Token.ID)        
        self.eat(Token.LPAR)
        self.function_name = function_name
        args = self.parse_arg_list()
        self.function_args = args
        self.eat(Token.RPAR)

    # You do not need to modify this for your homework
    # but you can look :) 
    def parse_arg_list(self) -> List[Tuple[str, str]]:
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.RPAR:
            return
        arg = self.parse_arg()
        token_id = self.get_token_id(self.to_match)
        if token_id == Token.RPAR:
            return [arg]
        self.eat(Token.COMMA)
        arg_l = self.parse_arg_list()
        return arg_l + [arg]

    # You do not need to modify this for your homework
    # but you can look :) 
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
            raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.INT, Token.FLOAT])
        self.eat(Token.AMP)
	# change strings and indexing token.names .value .token
        id_name = self.to_match.value
        self.eat(Token.ID)

        # storing an IO variable to the symbol table
        self.symbol_table.insert(id_name, IDType.IO, data_type)
        return (id_name, data_type_str)
        
    # The top level parsing function for your homework
    # This function needs to return a list of three address codes
    def parse_statement_list(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT, Token.FLOAT, Token.ID, Token.IF, Token.LBRACE, Token.FOR]:
            self.parse_statement()
            self.parse_statement_list()
            return []
        if token_id in [Token.RBRACE]:
            return []
        
    # you need to return a list of three address instructions
    # from the statement that gets parsed
    def parse_statement(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT, Token.FLOAT]:
            self.parse_declaration_statement()
        elif token_id in [Token.ID]:
            self.parse_assignment_statement()
        elif token_id in [Token.IF]:
            self.parse_if_else_statement()
        elif token_id in [Token.LBRACE]:
            self.parse_block_statement()
        elif token_id in [Token.FOR]:
            self.parse_for_statement()
        else:
            raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.FOR, Token.IF, Token.LBRACE, Token.INT, Token.FLOAT, Token.ID])

    # you need to return a list of three address instructions
    def parse_declaration_statement(self) -> List[str]:
        token_id = self.get_token_id(self.to_match)
        if token_id in [Token.INT]:
            self.eat(Token.INT)
            id_name = self.to_match.value
            # Think about what you want to insert into the symbol table
            # self.symbol_table.insert(...)
            self.eat(Token.ID)
            self.eat(Token.SEMI)
            return
        if token_id in [Token.FLOAT]:
            self.eat(Token.FLOAT)
            id_name = self.to_match.value
            # Think about what you want to insert into the symbol table
            # self.symbol_table.insert(...)
            self.eat(Token.ID)
            self.eat(Token.SEMI)
            return
        
        raise ParserException(self.scanner.get_lineno(),
                              self.to_match,            
                              [Token.INT, Token.FLOAT])

    # you need to return a list of three address instructions
    def parse_assignment_statement(self) -> List[str]:
        self.parse_assignment_statement_base()
        self.eat(Token.SEMI)
        return

    # you need to return a list of three address instructions
    def parse_assignment_statement_base(self) -> List[str]:
        id_name = self.to_match.value
        id_data = self.symbol_table.lookup(id_name)
        if id_data == None:
            raise SymbolTableException(self.scanner.get_lineno(), id_name)
        self.eat(Token.ID)
        self.eat(Token.ASSIGN)
        self.parse_expr()
        return 

    # you need to return a list of three address instructions
    def parse_if_else_statement(self) -> List[str]:
        self.eat(Token.IF)
        self.eat(Token.LPAR)
        self.parse_expr()
        self.eat(Token.RPAR)
        self.parse_statement()
        self.eat(Token.ELSE)
        self.parse_statement()
        return
    
    # you need to return a list of three address instructions
    def parse_block_statement(self) -> List[str]:
        self.eat(Token.LBRACE)
        self.symbol_table.push_scope()
        self.parse_statement_list()
        self.symbol_table.pop_scope()
        self.eat(Token.RBRACE)
        return

    # you need to return a list of three address instructions
    def parse_for_statement(self) -> List[str]:
        self.eat(Token.FOR)
        self.eat(Token.LPAR)
        self.parse_assignment_statement()
        self.parse_expr()
        self.eat(Token.SEMI)
        self.parse_assignment_statement_base()
        self.eat(Token.RPAR)
        self.parse_statement()
        return

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
