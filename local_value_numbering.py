# # import re

# # # perform the local value numbering optimization
# # def LVN(program):

# #     # returns 3 items:
    
# #     # 1. a new program (list of classier instructions)
# #     # with the LVN optimization applied

# #     # 2. a list of new variables required (e.g. numbered virtual
# #     # registers and program variables)

# #     # 3. a number with how many instructions were replaced    
# #     return program,[],0

# import re
# from typing import List, Tuple, Dict

# # Check if a variable is a virtual register
# def is_virtual_register(v: str) -> bool:
#     return v.startswith("vr") or v.startswith("_new_name")

# # Split the program into basic blocks
# def split_into_basic_blocks(program: List[str]) -> List[List[str]]:
#     blocks = []
#     current = []

#     for instr in program:
#         if re.match(r"^\w+:", instr):  # label starts a new block
#             if current:
#                 blocks.append(current)
#                 current = []
#         current.append(instr)
#         if any(instr.startswith(b) for b in ["br", "bne", "beq", "jmp"]):
#             blocks.append(current)
#             current = []

#     if current:
#         blocks.append(current)

#     return blocks

# # Perform local value numbering optimization
# def LVN(program: List[str]) -> Tuple[List[str], List[str], int]:
#     blocks = split_into_basic_blocks(program)
#     final_program = []
#     new_variables = []
#     replaced_count = 0
#     temp_counter = 0  # for generating new temp variables

#     for block in blocks:
#         value_table: Dict[Tuple[str, str, str], str] = {}  # maps (op, arg1, arg2) to virtual reg
#         var_value_number: Dict[str, int] = {}              # maps variable → value number
#         value_number_table: Dict[int, str] = {}            # maps value number → variable name
#         value_number_counter = 1
#         optimized_block = []

#         for instr in block:
#             instr = instr.strip()

#             # Match 2-operand arithmetic instructions: x = op(a, b)
#             match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\)", instr)
#             if match:
#                 dst, op, a1, a2 = match.groups()
#                 if all(is_virtual_register(arg) for arg in [a1, a2]) and op in ["add", "mul", "eq"]:
#                     key = (op, *sorted([a1, a2]))  # commutative ops

#                     if key in value_table:
#                         existing_var = value_table[key]
#                         optimized_block.append(f"{dst} = {existing_var}")
#                         replaced_count += 1
#                     else:
#                         value_table[key] = dst
#                         optimized_block.append(instr)
#                 else:
#                     optimized_block.append(instr)
#             else:
#                 # Non-matching instructions are kept as is
#                 optimized_block.append(instr)

#         final_program.extend(optimized_block)

#     return final_program, new_variables, replaced_count

import re
from typing import List, Tuple, Dict

def is_virtual_register(v: str) -> bool:
    return v.startswith("vr") or v.startswith("_new_name")

def split_into_basic_blocks(program: List[str]) -> List[List[str]]:
    blocks = []
    current = []
    for instr in program:
        if re.match(r"^\w+:", instr):  # label
            if current:
                blocks.append(current)
            current = [instr]
        else:
            current.append(instr)
            if any(instr.strip().startswith(prefix) for prefix in ["br", "bne", "beq"]):
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return blocks

def LVN(program: List[str]) -> Tuple[List[str], List[str], int]:
    blocks = split_into_basic_blocks(program)
    new_program = []
    new_vars = []
    replaced = 0

    for block in blocks:
        value_table: Dict[Tuple[str, str, str], str] = {}
        optimized = []

        for instr in block:
            instr = instr.strip()

            # Match binary operations: x = op(a, b)
            match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\);", instr)
            if match:
                dst, op, a, b = match.groups()
                if is_virtual_register(a) and is_virtual_register(b) and op in ["add", "mul", "eq"]:
                    key = (op, *sorted([a, b]))  # commutative key
                    if key in value_table:
                        # Replace with copy
                        prev_dst = value_table[key]
                        optimized.append(f"{dst} = {prev_dst};")
                        replaced += 1
                    else:
                        value_table[key] = dst
                        optimized.append(instr)
                else:
                    optimized.append(instr)
            else:
                optimized.append(instr)

        new_program.extend(optimized)

    return new_program, new_vars, replaced

