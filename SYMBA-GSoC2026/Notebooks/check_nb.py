import json

for name in ['Preprocess.ipynb', 'Training.ipynb', 'Eval_and_inference.ipynb']:
    with open(name, encoding='utf-8') as f:
        nb = json.load(f)
    print(f'\n=== {name} ===')
    print(f'  nbformat: {nb.get("nbformat")}.{nb.get("nbformat_minor")}')
    print(f'  cells: {len(nb["cells"])}')
    for i, c in enumerate(nb['cells']):
        ctype = c['cell_type']
        src = ''.join(c['source'])
        preview = src[:60].replace('\n', ' ')
        has_outputs = 'outputs' in c
        has_exec = 'execution_count' in c
        issues = []
        if ctype == 'code' and not has_outputs:
            issues.append('MISSING outputs')
        if ctype == 'code' and not has_exec:
            issues.append('MISSING execution_count')
        issue_str = f' !! {", ".join(issues)}' if issues else ''
        print(f'  [{i}] {ctype:8s} | {preview}...{issue_str}')
