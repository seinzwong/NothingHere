"""Exercise the production lock and atomic JSON writer without loading CUDA."""
import ast
import json
import os
import tempfile
import threading
from pathlib import Path
import msvcrt

tree = ast.parse(Path('full_validation_pipeline.py').read_text(encoding='utf-8'))
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('main', 'write_json')]
with tempfile.TemporaryDirectory(dir='full_validation_study') as folder:
    calls = []
    namespace = dict(Path=Path, os=os, json=json, threading=threading,
                     FULL=Path(folder), run_cli=lambda: calls.append('run'))
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<production-functions>', 'exec'), namespace)
    path = Path(folder) / 'state.json'
    path.write_bytes(b'\x00' * 32)
    namespace['write_json'](path, {'complete': True})
    assert json.loads(path.read_text()) == {'complete': True}
    namespace['main']()
    assert calls == ['run']
    with (Path(folder) / 'pipeline.lock').open('r+b') as held:
        msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, 1)
        namespace['main']()
        assert calls == ['run'], 'A duplicate run entered the pipeline'
        held.seek(0)
        msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
    namespace['main']()
    assert calls == ['run', 'run'], 'Lock was not released'
print('PASS: corrupt JSON replacement, exclusive run lock, lock release')
