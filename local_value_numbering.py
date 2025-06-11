
# local_value_numbering.py
import re
from typing import List, Tuple, Dict

def is_virtual_register(v: str) -> bool:
    return v.startswith("vr") or v.startswith("_new_name")

def split_into_basic_blocks(program: List[str]) -> List[List[str]]:
    blocks = []
    current = []
    for instr in program:
        if re.match(r"^\w+:", instr):
            if current:
                blocks.append(current)
            current = [instr]
        else:
            current.append(instr)
            if any(instr.strip().startswith(p) for p in ["br", "bne", "beq"]):
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
        optimized: List[str] = []
        for instr in block:
            instr = instr.strip()
            match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\);?", instr)
            if match:
                dst, op, a, b = match.groups()
                if is_virtual_register(a) and is_virtual_register(b) and op in ["add", "mul", "eq"]:
                    key = (op, *sorted([a, b]))
                    if key in value_table:
                        prev = value_table[key]
                        optimized.append(f"{dst} = {prev};")
                        replaced += 1
                    else:
                        value_table[key] = dst
                        optimized.append(instr if instr.endswith(';') else instr+';')
                else:
                    optimized.append(instr if instr.endswith(';') else instr+';')
            else:
                optimized.append(instr)
        new_program.extend(optimized)
    return new_program, new_vars, replaced

