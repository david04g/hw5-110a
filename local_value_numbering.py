import re
from typing import List, Tuple

def is_virtual(var: str) -> bool:
    return var.startswith("vr") or var.startswith("_new_name")

# Patterns for infix and functional operations, and copy
_infix_pat = re.compile(r"(\S+)\s*=\s*(\S+)\s*([+\-*/])\s*(\S+)")
_op_pat    = re.compile(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\)")
_copy_pat  = re.compile(r"(\S+)\s*=\s*(\S+)")

# Map infix operator symbols to functional names
_op_map = {
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div'
}

def parse_instruction(instr: str) -> Tuple[str, ...]:
    s = instr.strip()
    # Infix form: x = a + b
    m = _infix_pat.match(s)
    if m:
        dst, a1, sym, a2 = m.groups()
        op = _op_map[sym]
        return ('op', dst, op, a1, a2)
    # Functional form: x = op(a, b)
    m = _op_pat.match(s)
    if m:
        dst, op, a1, a2 = m.groups()
        return ('op', dst, op, a1, a2)
    # Copy form: x = y
    m = _copy_pat.match(s)
    if m:
        dst, src = m.groups()
        return ('copy', dst, src)
    return ('other', instr)

def LVN(program: List[str]) -> Tuple[List[str], List[str], int]:
    """
    Perform local value numbering on a list of three-address instructions.
    Returns (new_program, sorted new virtual regs, replaced_count).
    """
    expr_table   = {}
    new_program  = []
    new_vars     = set()
    replaced_cnt = 0

    for line in program:
        kind = parse_instruction(line)

        if kind[0] == 'op':
            _, dst, op, a1, a2 = kind
            if is_virtual(a1) and is_virtual(a2):
                # commutative operations
                if op in ('add', 'mul', 'eq'):
                    key = (op, tuple(sorted((a1, a2))))
                else:
                    key = (op, (a1, a2))

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

        # Other instructions are passed through
        new_program.append(line)

    return new_program, sorted(new_vars), replaced_cnt

