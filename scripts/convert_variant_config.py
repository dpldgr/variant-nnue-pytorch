#!/usr/bin/env python3
"""Convert generated variant.py + variant.h to the agreed JSON configuration."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from variant import VariantConfig


def convert(python_file,header_file):
    values={}
    def evaluate(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name):return values[node.id]
        if isinstance(node,ast.Dict):return {evaluate(k):evaluate(v) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.BinOp):
            a,b=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node,ast.IfExp):return evaluate(node.body if evaluate(node.test) else node.orelse)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -evaluate(node.operand)
        raise ValueError('Unsupported expression in generated Python configuration')
    for statement in ast.parse(Path(python_file).read_text()).body:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            values[statement.targets[0].id]=evaluate(statement.value)
        else:raise ValueError('Expected generated configuration assignments only')
    match=re.search(r'^\s*#define\s+PIECE_COUNT\s+(\d+)\s*$',Path(header_file).read_text(),re.M)
    if not match:raise ValueError('Missing numeric PIECE_COUNT in header')
    fields=('RANKS','FILES','KING_SQUARES','PIECE_TYPES','USE_POCKETS')
    result={name:values[name] for name in fields}
    result['PIECE_COUNT']=int(match.group(1))
    result['PIECE_VALUES']={str(k):v for k,v in values['PIECE_VALUES'].items()}
    return VariantConfig.from_dict(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('python_file');p.add_argument('header_file');p.add_argument('output')
    a=p.parse_args()
    Path(a.output).write_text(json.dumps(convert(a.python_file,a.header_file).to_dict(),indent=2)+'\n')

if __name__=='__main__':main()
