"""Compare generic outputs with independently compiled prp_data_loader DLLs.

Usage: python tests/compare_loaders.py /path/to/compiled/reference/profiles
Each profile folder contains libtraining_data_loader.so, variant.json, positions.bin.
"""
import ctypes
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import nnue_dataset as data
import variant
from generic_fixtures import PROFILES,write


def bind(lib,name,result,args):
    f=getattr(lib,name);f.restype=result;f.argtypes=args;return f


class Reference:
    def __init__(self,folder,feature,filename,shuffle,workers):
        self.dll=ctypes.CDLL(str(folder/'libtraining_data_loader.so'))
        common=[ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_bool,ctypes.c_bool,ctypes.c_int]
        if shuffle:
            create=bind(self.dll,'create_prp_sparse_batch_stream',ctypes.c_void_p,common+[ctypes.c_uint64,ctypes.c_uint64])
            self.fetch=bind(self.dll,'fetch_next_prp_sparse_batch',data.SparseBatchPtr,[ctypes.c_void_p])
            extra=(17,3)
        else:
            create=bind(self.dll,'create_sparse_batch_stream',ctypes.c_void_p,common)
            self.fetch=bind(self.dll,'fetch_next_sparse_batch',data.SparseBatchPtr,[ctypes.c_void_p]);extra=()
        self.free=bind(self.dll,'destroy_sparse_batch',None,[data.SparseBatchPtr])
        self.close_fn=bind(self.dll,'destroy_sparse_batch_stream',None,[ctypes.c_void_p])
        self.handle=create(feature.encode(),workers,str(filename).encode(),32,False,False,0,*extra)
        assert self.handle
    def next(self):
        b=self.fetch(self.handle)
        if not b:return None
        try:return b.contents.get_tensors('cpu')
        finally:self.free(b)
    def close(self):self.close_fn(self.handle)


def main():
    root=Path(sys.argv[1]);cases=[]
    for name,cfg in PROFILES.items():
        folder=root/name;config=variant.VariantConfig.from_dict(cfg)
        for feature in ('HalfKP','HalfKP^','HalfKA','HalfKA^','HalfKAv2','HalfKAv2^'):
            if name=='no-reference' and feature.startswith('HalfKP'):continue # pre-existing unsupported reference-less HalfKP factorization
            for shuffle in (False,True):
                workers=4 if shuffle else 1
                reference=Reference(folder,feature,folder/'positions.bin',shuffle,workers)
                cls=data.PrpSparseBatchProvider if shuffle else data.SparseBatchProvider
                kwargs=dict(config=config,num_workers=workers)
                if shuffle:kwargs.update(seed=17,epoch=3)
                else:kwargs.update(cyclic=False)
                provider=cls(feature,str(folder/'positions.bin'),32,**kwargs)
                count=0
                try:
                    while True:
                        expected=reference.next()
                        try:actual=next(provider)
                        except StopIteration:actual=None
                        assert (expected is None)==(actual is None),(name,feature,shuffle,'batch count')
                        if actual is None:break
                        for index,(a,b) in enumerate(zip(actual,expected)):
                            torch.testing.assert_close(a,b,rtol=0,atol=0,msg=f'{name} {feature} shuffle={shuffle} field={index}')
                        count+=len(actual[7])
                finally:provider.close();reference.close()
                assert count==137
                cases.append(dict(profile=name,feature=feature,prp=shuffle,records=count))
        print(name,': native tensors match compiled reference',flush=True)
    (root.parent/'reference-parity.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(len(cases),'complete native comparisons passed',flush=True)

if __name__=='__main__':main()
