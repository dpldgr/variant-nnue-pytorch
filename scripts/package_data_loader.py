#!/usr/bin/env python3
"""Bundle a prebuilt generic loader and its default/example configuration."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    files=[(args.library,args.library.name),(root/'variant.json','variant.json'),
           (root/'configs/metal-chess.json','configs/metal-chess.json')]
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path,name in files:archive.write(path,name)
        archive.writestr('README.txt',
            'Generic data-loader ABI 2. Extract the native library next to nnue_dataset.py.\n'
            'No per-variant compiler is required. Use --variant-config PATH to select JSON.\n'
            'Keep a configured variant.json if you already have one.\n')
        archive.writestr('SHA256SUMS',''.join(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+name+'\n' for path,name in files))

if __name__=='__main__':main()
