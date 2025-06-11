import re

def is_virtual(var):
    return var.startswith("vr") or var.startswith("_new_name")

def parse_instruction(instr):
    instr = instr.strip()
    m = re.match(r"(\S+)\s*=\s*(\S+)\((\S+),\s*(\S+)\)", instr)
    if m:
        dest, op, arg1, arg2 = m.groups()
        return ('op', dest, op, arg1, arg2)
    m = re.match(r"(\S+)\s*=\s*(\S+)", instr)
    if m:
        dest, src = m.groups()
        return ('copy', dest, src)
    return ('other', instr)

def LVN(program):
    expr_table = {}
    value_table = {}
    new_program = []
    new_vars = set()
    replaced = 0

    for line in program:
        kind = parse_instruction(line)

        if kind[0] == 'op':
            _, dest, op, arg1, arg2 = kind
            if not (is_virtual(arg1) and is_virtual(arg2)):
                new_program.append(line)
                continue

            key = (op, tuple(sorted([arg1, arg2])) if op in ['add', 'mul', 'eq'] else (arg1, arg2))
            if key in expr_table:
                existing = expr_table[key]
                new_program.append(f"{dest} = {existing}")
                replaced += 1
            else:
                expr_table[key] = dest
                new_program.append(line)
            new_vars.add(dest)

        elif kind[0] == 'copy':
            _, dest, src = kind
            new_program.append(line)
            if is_virtual(dest):
                new_vars.add(dest)

        else:
            new_program.append(line)

    return new_program, sorted(list(new_vars)), replaced
