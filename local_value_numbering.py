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
    expr_table   = {}
    new_program  = []
    new_vars     = set()
    replaced_cnt = 0

    for line in program:
        stripped = line.strip()

        # 1) detect label or branch → start new basic block
        if stripped.endswith(':') or stripped.startswith('goto ') or stripped.startswith('ifFalse '):
            expr_table.clear()
            new_program.append(line)
            continue

        kind = parse_instruction(line)

        # 2) arithmetic operations
        if kind[0] == 'op':
            _, dst, op, a1, a2 = kind
            if is_virtual(a1) and is_virtual(a2):
                key = (op, tuple(sorted((a1, a2)))) if op in ('add','mul','eq') else (op, (a1,a2))
                if key in expr_table:
                    new_program.append(f"{dst} = {expr_table[key]}")
                    replaced_cnt += 1
                else:
                    expr_table[key] = dst
                    new_program.append(line)
                new_vars.add(dst)
                continue

        # 3) copy instructions
        if kind[0] == 'copy':
            _, dst, src = kind
            new_program.append(line)
            if is_virtual(dst):
                new_vars.add(dst)
            continue

        # 4) any other instruction (e.g. I/O, declarations) ends the block
        expr_table.clear()
        new_program.append(line)

    return new_program, sorted(new_vars), replaced_cnt

