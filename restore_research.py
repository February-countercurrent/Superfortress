"""Expand full research source into research/workspace without changing the two main agents."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent

def write_checked(destination,data,expected):
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Archive hash mismatch')
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:
            raise FileExistsError('Refusing to replace modified file: '+str(destination))
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as handle:handle.write(data)

def main():
    index=json.loads((ROOT/'research/index.json').read_text(encoding='utf-8'))
    target=(ROOT/'research/workspace').resolve()
    with zipfile.ZipFile(ROOT/index['archive']) as archive:
        for row in index['files']:
            destination=(target/row['path']).resolve()
            if not destination.is_relative_to(target):raise ValueError('Invalid archive path')
            write_checked(destination,archive.read(row['path']),row['sha256'])
    for relative,source in index['provided_model_copies'].items():
        destination=(target/relative).resolve()
        if not destination.is_relative_to(target):raise ValueError('Invalid model destination')
        data=(ROOT/source).read_bytes()
        write_checked(destination,data,hashlib.sha256(data).hexdigest())
    for name in ['logs','replays','results','screenshots']:(target/name).mkdir(exist_ok=True)
    print(f'Restored {len(index["files"])} files to {target}')
    print('Historical weights and raw training data are not bundled. See research/README.md.')

if __name__=='__main__':main()
