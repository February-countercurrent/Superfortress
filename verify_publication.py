"""Check the two-model publication and its complete research source archive."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=ROOT/row['path']
        assert path.is_file(),row['path']
        assert digest(path.read_bytes())==row['sha256'],row['path']
    for rel,expected in manifest['protected_artifacts'].items():
        assert digest((ROOT/rel).read_bytes())==expected,rel
    index=json.loads((ROOT/'research/index.json').read_text(encoding='utf-8'))
    python_count=0
    with zipfile.ZipFile(ROOT/index['archive']) as archive:
        for row in index['files']:
            data=archive.read(row['path'])
            assert digest(data)==row['sha256'],row['path']
            if row['path'].endswith('.py'):
                ast.parse(data.decode('utf-8-sig'),filename=row['path']);python_count+=1
    with zipfile.ZipFile(ROOT/'release/final-project-agent-code.zip') as archive:
        for info in archive.infolist():
            if info.is_dir():continue
            assert (ROOT/'agent_code'/info.filename).read_bytes()==archive.read(info),info.filename
    print(f'OK: {len(manifest["files"])} publication files, {python_count} archived Python files, both trained models and the exact submission.')

if __name__=='__main__':main()
