# local_value_numbering.py
import re
from typing import List, Tuple, Dict, Set

def is_virtual_register(v: str) -> bool:
    """Checks if a string is a virtual register."""
    return v.startswith("vr") or v.startswith("_new_name")

def is_memory_variable(v: str) -> bool:
    """Checks if a string is a memory variable (i.e., not a VR or literal)."""
    # Handles cases like `x` but not `"some_string"` or integer literals
    return v.isidentifier() and not is_virtual_register(v)

def split_into_basic_blocks(program: List[str]) -> List[List[str]]:
    """Splits a list of instructions into basic blocks."""
    blocks = []
    current_block = []
    for instr in program:
        stripped_instr = instr.strip()
        if not stripped_instr:
            continue
        
        if re.match(r"^\w+:", stripped_instr):
            if current_block:
                blocks.append(current_block)
            current_block = [instr]
        else:
            current_block.append(instr)
            if any(stripped_instr.startswith(p) for p in ["br", "bne", "beq", "jmp"]):
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)
    return blocks

def LVN(program: List[str]) -> Tuple[List[str], List[str], int]:
    """Performs local value numbering with corrected invalidation logic."""
    blocks = split_into_basic_blocks(program)
    new_program: List[str] = []
    replaced_count = 0
    
    COMMUTATIVE_OPS: Set[str] = {"add", "mul", "eq", "addi", "addf"}
    NON_COMMUTATIVE_OPS: Set[str] = {"lti", "subi", "subf", "divi", "divf"}
    # Note: load/store ops are handled separately
    UNARY_OPS: Set[str] = {"vr_int2float"} 

    for block in blocks:
        value_table: Dict[Tuple[str, ...], str] = {}
        var_map: Dict[str, str] = {}
        optimized_block: List[str] = []

        for instr in block:
            instr_str = instr.strip()

            # --- 1. Parse Instruction and Create Value Key ---
            key = None
            dst = None
            is_copy = False
            is_store = False

            binary_match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\);?", instr_str)
            unary_match = re.match(r"(\S+)\s*=\s*(\S+)\((\S+)\);?", instr_str)
            copy_match = re.match(r"(\S+)\s*=\s*(\S+);", instr_str)
            
            if binary_match:
                dst, op, a, b = binary_match.groups()
                canon_a = var_map.get(a, a)
                canon_b = var_map.get(b, b)
                if op in COMMUTATIVE_OPS: key = (op, *sorted([canon_a, canon_b]))
                elif op in NON_COMMUTATIVE_OPS: key = (op, canon_a, canon_b)
            
            elif unary_match:
                dst, op, a = unary_match.groups()
                canon_a = var_map.get(a, a)
                # Check for load (e.g., vr = float2vr(x)) or literal assignment (vr = int2vr(1))
                load_match = re.match(r"(float|int)2vr", op)
                if load_match:
                    if is_memory_variable(a):
                        key = ('load', a)
                    else: # It's a literal assignment, e.g., int2vr(1)
                        key = (op, a)
                elif op in UNARY_OPS:
                    key = (op, canon_a)
            
            elif copy_match:
                dst, _ = copy_match.groups()
                is_copy = True

            if not dst and re.match(r"(\S+)\s*=\s*vr2(float|int)", instr_str):
                 is_store = True
                 dst = instr_str.split('=')[0].strip()

            # --- 2. Attempt Optimization ---
            optimized = False
            if key and key in value_table:
                prev_dst = value_table[key]
                optimized_block.append(f"  {dst} = {prev_dst};")
                var_map[dst] = var_map.get(prev_dst, prev_dst) # Update alias map
                replaced_count += 1
                optimized = True

            # --- 3. Invalidate Destination (AFTER optimization check) ---
            if dst:
                # Invalidate any value that `dst` previously held
                keys_to_del = {k for k, v in value_table.items() if v == dst}
                for k in keys_to_del: del value_table[k]

                # Remove old aliases
                if dst in var_map: del var_map[dst]
                for k, v in list(var_map.items()):
                    if v == dst: del var_map[k]
                
                # If `dst` is a memory location (from a store), invalidate our knowledge of it
                if is_store and is_memory_variable(dst):
                    if ('load', dst) in value_table:
                        del value_table[('load', dst)]

            # --- 4. Update Tables if No Optimization Occurred ---
            if not optimized:
                optimized_block.append(instr)
                if key:
                    value_table[key] = dst
                elif is_copy:
                    dst, src = copy_match.groups()
                    var_map[dst] = var_map.get(src, src)
            
            # A store always updates our knowledge of what's in memory
            store_match = re.match(r"(\S+)\s*=\s*vr2(float|int)\((\S+)\);?", instr_str)
            if store_match:
                mem_loc, _, src_reg = store_match.groups()
                canon_src = var_map.get(src_reg, src_reg)
                value_table[('load', mem_loc)] = canon_src

        new_program.extend(optimized_block)
        
    return new_program, [], replaced_count

