import re
from typing import List, Tuple

def is_virtual(var: str) -> bool:
    return var.startswith("vr") or var.startswith("_new_name")

# Patterns for matching ClassIeR instructions
_op_pat   = re.compile(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\)")
_copy_pat = re.compile(r"(\S+)\s*=\s*(\S+)")

def parse_instruction(instr: str) -> Tuple[str, ...]:
    instr = instr.strip()
    m = _op_pat.match(instr)
    if m:
        dst, op, a1, a2 = m.groups()
        return ('op', dst, op, a1, a2)
    m = _copy_pat.match(instr)
    if m:
        dst, src = m.groups()
        return ('copy', dst, src)
    return ('other', instr)

def LVN(program: List[str]) -> Tuple[List[str], List[str], int]:
    """
    Perform local value numbering:
    - program: list of ClassIeR code lines
    Returns:
      1. new program list with arithmetic redundancies replaced by copies
      2. list of new virtual registers used
      3. count of replaced arithmetic instructions
    """
    expr_table   = {}             # maps (op, operands) -> existing dest
    new_program  = []             # output lines
    new_vars     = set()          # collect any new regs
    replaced_cnt = 0

    for line in program:
        kind = parse_instruction(line)

        if kind[0] == 'op':
            _, dst, op, a1, a2 = kind
            # only virtual registers
            if is_virtual(a1) and is_virtual(a2):
                # commutative ops
                if op in ('add','mul','eq'):
                    key = (op, tuple(sorted((a1,a2))))
                else:
                    key = (op, (a1,a2))

                if key in expr_table:
                    existing = expr_table[key]
                    new_program.append(f"{dst} = {existing}")
                    replaced_cnt += 1
                else:
                    expr_table[key] = dst
                    new_program.append(line)
                new_vars.add(dst)
                continue
        if kind[0] == 'copy':
            _, dst, src = kind
            new_program.append(line)
            if is_virtual(dst):
                new_vars.add(dst)
            continue

        # all other instructions
        new_program.append(line)

    return new_program, sorted(new_vars), replaced_cnt

