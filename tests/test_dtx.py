"""DTX labels/feature parity, all reader strategies and malformed file checks."""
import pathlib
import tempfile
import unittest
import torch
import variant
import nnue_dataset as data
from dtx_fixture import write,frame,header,labels
from test_record_bytes import frame as bin_frame


class DtxTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name)
        distances=(-1024,0,1,-1,2,-2,1023,-1023)
        self.rows=[((0,63),[(1+(i*7)%31,(i+j)%5,(i+j)%2) if j==0 else (32+(i*11)%31,(i+j)%5,(i+j)%2) for j in range(i%3)],i%2,distances[i%8]) for i in range(137)]
        self.path=write(self.root/'input.dtx',self.rows)
        self.bin=self.root/'input.bin'
        self.bin.write_bytes(b''.join(bin_frame(k,p,t,*labels(d),72) for k,p,t,d in self.rows))
    def tearDown(self):self.temp.cleanup()
    def collect(self,path,mode='sequential',feature='HalfKAv2',**kwargs):
        common=dict(num_workers=kwargs.pop('num_workers',4),config=kwargs.pop('config',None),**kwargs)
        if mode=='sequential':provider=data.SparseBatchProvider(feature,str(path),32,cyclic=False,**common)
        else:provider=data.PrpSparseBatchProvider(feature,str(path),32,seed=17,epoch=2,prp_mode=mode,prp_block_size=17,prp_buffer_blocks=3,**common)
        try:
            batches=list(provider)
            return tuple(torch.cat([b[i] for b in batches],dim=0) for i in range(10))
        finally:provider.close()
    def test_all_features_and_strategies_match_bin(self):
        self.assertEqual(data.training_data_record_count(self.path),137)
        for f in ('HalfKP','HalfKP^','HalfKA','HalfKA^','HalfKAv2','HalfKAv2^'):
            for mode in ('sequential','global','buffered'):
                with self.subTest(feature=f,mode=mode):
                    expected=self.collect(self.bin,mode,f,**({'num_workers':1} if mode=='sequential' else {}))
                    actual=self.collect(self.path,mode,f)
                    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_header_controls_stride_and_partial_batches(self):
        expected=self.collect(self.path,'buffered')
        for width in (7,8,144):
            actual=self.collect(self.path,'buffered',record_bytes=width)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        p=data.PrpSparseBatchProvider('HalfKAv2',self.path,32)
        try:self.assertEqual([len(b[7]) for b in p],[32,32,32,32,9])
        finally:p.close()
    def test_resume_after_lossless_format_conversion(self):
        before=data.PrpBatchDataset('HalfKAv2',str(self.bin),32,seed=17)
        list(before);state=before.state_dict()
        after=data.PrpBatchDataset('HalfKAv2',self.path,32,seed=17)
        after.load_state_dict(state)
        self.assertEqual(after.next_epoch,1)
        expected=list(before);actual=list(after)
        for a,b in zip(actual,expected):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    def test_rectangle_and_distance_widths(self):
        cfg=variant.VariantConfig.from_dict(dict(FILES=4,RANKS=3,PIECE_TYPES=6,PIECE_COUNT=12,KING_SQUARES=12,USE_POCKETS=False,PIECE_VALUES={}))
        rows=[((0,11),[(3,0,0),(7,4,1)],i%2,d) for i,d in enumerate((-64,0,1,-1,63,-63))]
        path=write(self.root/'small.DTX',rows,files=4,ranks=3,variant='small',distance_bits=7)
        self.assertEqual(data.training_data_record_count(path,config=cfg),6)
        batch=self.collect(path,config=cfg)
        self.assertEqual(batch[7].flatten().tolist(),[labels(r[-1],7)[0] for r in rows])
        for bits in (1,7,11,15):
            rows=[((0,63),[],0,-(1<<(bits-1))),((0,63),[],1,0)]
            path=write(self.root/f'width-{bits}.dtx',rows,distance_bits=bits)
            self.assertEqual(self.collect(path)[7].flatten().tolist(),[0,-32000])
    def test_storage_capacity_does_not_change_model_piece_count(self):
        cfg=variant.VariantConfig.from_dict(dict(FILES=8,RANKS=8,PIECE_TYPES=6,PIECE_COUNT=2,KING_SQUARES=64,USE_POCKETS=False,PIECE_VALUES={}))
        path=write(self.root/'kings.dtx',[((0,63),[],0,-1024)])
        self.assertEqual(len(self.collect(path,config=cfg)[7]),1)
        path=write(self.root/'too-many.dtx',[((0,63),[(1,0,0)],0,1)])
        with self.assertRaisesRegex(RuntimeError,'PIECE_COUNT'):self.collect(path,config=cfg)
    def test_bad_headers_rejected(self):
        original=pathlib.Path(self.path).read_bytes()
        cases=[(0,255),(8,2),(10,0),(16,1),(20,7),(24,4),(26,33),(28,2),(29,1),(30,16),(31,2),(40,0),(48,1),(52,0),(53,1),(54,2),(56,1)]
        for offset,value in cases:
            with self.subTest(offset=offset):
                raw=bytearray(original);raw[offset]=value;p=self.root/'invalid.dtx';p.write_bytes(raw)
                with self.assertRaises(RuntimeError):data.training_data_record_count(str(p))
        for raw in (original[:63],original[:-1],original+b'x',header(0)[0]):
            p=self.root/'invalid.dtx';p.write_bytes(raw)
            with self.assertRaises(RuntimeError):data.training_data_record_count(str(p))
    def test_bad_payload_and_padding_rejected(self):
        head,stride,_,_=header(1)
        row=bytearray(frame((0,63),[],0,-1024))
        # Bad padding, overlapping Kings, nonzero empty slot, occupied slot after an empty.
        words=[int.from_bytes(row,'little')|(1<<44),0,int.from_bytes(row,'little')|(1<<12),int.from_bytes(row,'little')|(1<<(22+6))]
        for word in words:
            p=self.root/'bad.dtx';p.write_bytes(head+word.to_bytes(stride,'little'))
            with self.assertRaises(RuntimeError):self.collect(p,'buffered')


if __name__=='__main__':unittest.main()
