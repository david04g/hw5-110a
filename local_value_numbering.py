# local_value_numbering.py
import re
from typing import List, Tuple, Dict

def is_virtual_register(v: str) -> bool:
    """Checks if a string is a virtual register."""
    return v.startswith("vr") or v.startswith("_new_name")

def split_into_basic_blocks(program: List[str]) -> List[List[str]]:
    """Splits a list of instructions into basic blocks."""
    blocks = []
    current = []
    for instr in program:
        instr_strip = instr.strip()
        if not instr_strip:
            continue
        
        if re.match(r"^\w+:", instr_strip):
            if current:
                blocks.append(current)
            current = [instr]
        else:
            current.append(instr)
            if any(instr_strip.startswith(prefix) for prefix in ["br", "bne", "beq", "jmp"]):
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return blocks

def LVN(program: List[str]) -> Tuple[List[str], List[str], int]:
    """Performs local value numbering optimization on the given program."""
    blocks = split_into_basic_blocks(program)
    new_program: List[str] = []
    new_vars: List[str] = []
    replaced = 0

    for block in blocks:
        value_table: Dict[Tuple[str, ...], str] = {}
        var_map: Dict[str, str] = {}
        optimized: List[str] = []

        for instr in block:
            instr_str = instr.strip()

            # Invalidation on redefinition
            dst_match = re.match(r"^\s*(\S+)\s*=", instr_str)
            if dst_match:
                dst = dst_match.group(1)
                keys_to_remove = {k for k, v in value_table.items() if v == dst}
                keys_to_remove.update({k for k in value_table if dst in k[1:]})
                for k in keys_to_remove: del value_table[k]
                if dst in var_map: del var_map[dst]
                aliases_to_dst = [k for k, v in var_map.items() if v == dst]
                for k in aliases_to_dst: del var_map[k]

            # Optimization logic
            binary_op_match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\);?", instr_str)
            unary_op_match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+)\);?", instr_str)
            copy_match = re.match(r"(\S+)\s*=\s*(\S+);", instr_str)

            if binary_op_match:
                dst, op, a, b = binary_op_match.groups()
                canon_a = var_map.get(a, a)
                canon_b = var_map.get(b, b)
                
                commutative_ops = ["add", "mul", "eq", "addi", "addf"]
                non_commutative_ops = ["lti"] # Add other non-commutative ops here
                
                key = None
                if op in commutative_ops:
                    key = (op, *sorted([canon_a, canon_b]))
                elif op in non_commutative_ops:
                    key = (op, canon_a, canon_b)

                if key and key in value_table:
                    prev_dst = value_table[key]
                    optimized.append(f"{dst} = {prev_dst};")
                    var_map[dst] = prev_dst
                    replaced += 1
                else:
                    if key: value_table[key] = dst
                    optimized.append(instr)
            elif unary_op_match:
                dst, op, a = unary_op_match.groups()
                canon_a = var_map.get(a, a)
                
                optimizable_unary_ops = ["int2vr", "float2vr", "vr_int2float", "vr2float", "vr2int"]
                key = (op, canon_a)

                if op in optimizable_unary_ops and key in value_table:
                    prev_dst = value_table[key]
                    optimized.append(f"{dst} = {prev_dst};")
                    var_map[dst] = prev_dst
                    replaced += 1
                else:
                    if op in optimizable_unary_ops: value_table[key] = dst
                    optimized.append(instr)
            elif copy_match:
                dst, src = copy_match.groups()
                var_map[dst] = var_map.get(src, src)
                optimized.append(instr)
            else:
                optimized.append(instr)

        new_program.extend(optimized)
    return new_program, new_vars, replaced
