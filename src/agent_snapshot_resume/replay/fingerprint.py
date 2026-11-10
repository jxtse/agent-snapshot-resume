"""Task-root fingerprints without including runtime/credential directories.

build_fingerprint_command(roots, optional_roots) returns a `python3 -c` command that prints
`files=<n>` and a sha256 over (path, size, sha256) of every file under the given absolute roots.
The same command runs in the source sandbox before the snapshot and in every clone; the outputs must match.
"""
import json
import shlex

# Default when a bundle names no roots: the conventional task workdir.
DEFAULT_ROOTS = ("/app",)


def build_fingerprint_command(roots, optional_roots=()):
    if not roots or any(not p.startswith('/') or p in ('/', '/root') for p in roots):
        raise ValueError('Specify explicit absolute task-data/work/output roots, not whole home/root')
    if not set(optional_roots).issubset(roots):
        raise ValueError('Optional roots must be explicitly listed fingerprint roots')
    code = '''import hashlib,json,pathlib
roots=json.loads(ROOTS_JSON)
optional=json.loads(OPTIONAL_JSON)
files=[]
missing=[]
for root in roots:
    p=pathlib.Path(root)
    if not p.exists():
        if root not in optional: raise RuntimeError('Missing fingerprint root: '+root)
        missing.append([root,None,None])
        continue
    files.extend([p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()])
rows=[]
for p in sorted(set(files)):
    with p.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
    rows.append([str(p),p.stat().st_size,h])
payload=json.dumps(sorted(rows+missing,key=lambda row:row[0]),separators=(',',':')).encode()
print('files='+str(len(rows)))
print(hashlib.sha256(payload).hexdigest())
'''.replace('ROOTS_JSON',repr(json.dumps(roots))).replace('OPTIONAL_JSON',repr(json.dumps(list(optional_roots))))
    return 'python3 -c '+shlex.quote(code)
