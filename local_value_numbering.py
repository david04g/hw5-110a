import re

# perform the local value numbering optimization
def LVN(program):

    # returns 3 items:
    
    # 1. a new program (list of classier instructions)
    # with the LVN optimization applied

    # 2. a list of new variables required (e.g. numbered virtual
    # registers and program variables)

    # 3. a number with how many instructions were replaced    
    return program,[],0
