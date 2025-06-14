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
        if not instr_strip: # Skip empty or whitespace-only lines
            continue
            
        # A new basic block starts at a label.
        if re.match(r"^\w+:", instr_strip):
            if current:
                blocks.append(current)
            current = [instr]
        else:
            current.append(instr)
            # A basic block ends with a branch or jump instruction.
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
        # value_table maps a value (op, operands) to the variable that holds it.
        value_table: Dict[Tuple[str, ...], str] = {}
        # var_map tracks aliases, mapping a variable to its canonical name.
        var_map: Dict[str, str] = {}
        optimized: List[str] = []

        for instr in block:
            instr_str = instr.strip()

            # --- Invalidation on Redefinition ---
            dst_match = re.match(r"^\s*(\S+)\s*=", instr_str)
            if dst_match:
                dst = dst_match.group(1)
                
                # 1. Invalidate stale entries in value_table.
                # Remove entries where `dst` was the result or an operand.
                keys_to_remove = {k for k, v in value_table.items() if v == dst}
                keys_to_remove.update({k for k in value_table if dst in k[1:]})
                for k in keys_to_remove:
                    del value_table[k]
                
                # 2. Update alias information.
                # `dst` is being redefined, so it's no longer an alias for anything.
                if dst in var_map:
                    del var_map[dst]
                # Other variables that were aliased to `dst` are now orphaned.
                # We remove their alias entry, making them their own canonical name.
                aliases_to_dst = [k for k, v in var_map.items() if v == dst]
                for k in aliases_to_dst:
                    del var_map[k]

            # --- Optimization with Alias Tracking ---
            op_match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\);?", instr_str)
            copy_match = re.match(r"(\S+)\s*=\s*(\S+);", instr_str)

            if op_match:
                dst, op, a, b = op_match.groups()
                # Resolve operands to their canonical names using the var_map.
                canon_a = var_map.get(a, a)
                canon_b = var_map.get(b, b)

                if is_virtual_register(canon_a) and is_virtual_register(canon_b) and op in ["add", "mul", "eq"]:
                    key = (op, *sorted([canon_a, canon_b]))
                    if key in value_table:
                        prev_dst = value_table[key]
                        optimized.append(f"{dst} = {prev_dst};")
                        var_map[dst] = prev_dst # Record the new alias.
                        replaced += 1
                    else:
                        value_table[key] = dst
                        optimized.append(instr)
                else:
                    optimized.append(instr)
            elif copy_match:
                dst, src = copy_match.groups()
                # Record the new alias.
                var_map[dst] = var_map.get(src, src)
                optimized.append(instr)
            else:
                # For non-optimizable instructions like labels and branches, preserve them.
                optimized.append(instr)

        new_program.extend(optimized)
    return new_program, new_vars, replaced
