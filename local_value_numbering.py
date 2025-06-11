# lvn.py

import re
from typing import List, Tuple

def is_virtual(var: str) -> bool:
    return var.startswith("vr") or var.startswith("_new_name")

# match infix: x = a + b
_infix_pat = re.compile(r"(\S+)\s*=\s*(\S+)\s*([+\-*/])\s*(\S+)")
# match functional: x = op(a, b)
_op_pat    = re.compile(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\)")
# match copy:      x = y
_copy_pat  = re.compile(r"(\S+)\s*=\s*(\S+)")

_op_map = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}

def parse_instruction(instr: str) -> Tuple[str, ...]:
    s = instr.strip()
    m = _infix_pat.match(s)
    if m:
        dst, a1, sym, a2 = m.groups()
        return ('op', dst, _op_map[sym], a1, a2)
    m = _op_pat.match(s)
    if m:
        dst, op, a1, a2 = m.groups()
        return ('op', dst, op, a1, a2)
    m = _copy_pat.match(s)
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
        if stripped.endswith(':') or stripped.startswith('goto ') or stripped.startswith('ifFalse '):
            expr_table.clear()
            new_program.append(line)
            continue

        kind = parse_instruction(line)
        if kind[0] == 'op':
            _, dst, op, a1, a2 = kind
            if is_virtual(a1) and is_virtual(a2):
                key = (op, tuple(sorted((a1, a2)))) if op in ('add','mul','eq') else (op, (a1, a2))
                if key in expr_table:
                    new_program.append(f"{dst} = {expr_table[key]}")
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

        expr_table.clear()
        new_program.append(line)

    return new_program, sorted(new_vars), replaced_cnt

