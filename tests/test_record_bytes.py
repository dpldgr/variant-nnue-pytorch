"""Independent wire encoders test native decoding and both loading strategies."""
import struct
import tempfile
import pathlib
import unittest
import torch
import nnue_dataset as data


def frame(kings, pieces, turn, score, result, size, cells=64, types=5):
    # Independent integer-based test encoder; not the production C++ codec.
    word=0; at=0
    def put(value,bits):
        nonlocal word,at
        word|=(value&((1<<bits)-1))<<at;at+=bits
    if size==72:
        if cells!=64 or types!=5:raise ValueError('test packed fixture schema')
        put(turn,1);put(kings[0],7);put(kings[1],7)
        by_square={p[0]:p for p in pieces}
        for rank in range(7,-1,-1):
            for file in range(8):
                sq=rank*8+file
                if sq in kings:continue
                if sq not in by_square:put(0,1)
                else:
                    _,kind,color=by_square[sq];put(2*kind+1,5);put(color,1)
        put(0,60);put(0,4);put(0,1);put(0,6);put(1,16);put(0,1)
        assert at<=512
        return word.to_bytes(64,'little')+struct.pack('<hHHbB',score,0,1,result,0)
    sb=max(6,(cells-1).bit_length());tb=max(3,(types-1).bit_length())
    cap=0
    for n in range(1,size*8+1):
        if 2*sb+n*(sb+tb+1)+n.bit_length()+20>size*8:break
        cap=n
    bitmap=len(pieces)>cap
    if bitmap:
        pb=types.bit_length()
        if cells+(len(pieces)+2)*(pb+1)+20>size*8:raise ValueError('does not fit')
        occupied={p[0]:(p[1],p[2]) for p in pieces}
        occupied[kings[0]]=(types,0);occupied[kings[1]]=(types,1)
        put(sum(1<<sq for sq in occupied),cells)
        for sq in sorted(occupied):put(occupied[sq][0],pb);put(occupied[sq][1],1)
    else:
        put(kings[0],sb);put(kings[1],sb)
        for i in range(cap):
            sq,kind,color=pieces[i] if i<len(pieces) else (0,0,0)
            put(sq,sb);put(kind,tb);put(color,1)
    put(turn,1)
    if not bitmap:put(len(pieces),cap.bit_length())
    put(score,16);put(result+1,2)
    assert at<size*8
    if bitmap:word|=1<<(size*8-1)
    return word.to_bytes(size,'little')


def rows(count):
    for i in range(count):
        kings=(0,63)
        squares=[1+(i*7)%31,32+(i*11)%31]
        supports=[(squares[j],(i+j)%5,(i+j)%2) for j in range(i%3)]
        yield kings,supports,i%2,i-100,(i%3)-1


class RecordBytesTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name)
        self.positions=list(rows(137))

    def tearDown(self):self.temp.cleanup()

    def write(self,size,positions=None):
        path=self.root/f'input-{size}.bin'
        path.write_bytes(b''.join(frame(*r,size) for r in (self.positions if positions is None else positions)))
        return str(path)

    def provider(self,path,size,shuffle=False,**kwargs):
        if shuffle:return data.PrpSparseBatchProvider('HalfKAv2',path,32,record_bytes=size,**kwargs)
        return data.SparseBatchProvider('HalfKAv2',path,32,cyclic=False,record_bytes=size,**kwargs)

    def collect(self,provider):
        try:
            batches=list(provider)
            return tuple(torch.cat([b[i] for b in batches],dim=0) for i in range(10)) if batches else ()
        finally:provider.close()

    def test_every_integer_width_decodes_like_packed(self):
        baseline=self.collect(self.provider(self.write(72),72))
        for size in range(7,145):
            with self.subTest(size=size):
                path=self.write(size)
                self.assertEqual(data.training_data_record_count(path,size),137)
                # Legacy PackedSfen sequential workers can finish batches out of
                # order. Compare physical record order with one legacy worker;
                # retain four workers for every compact width.
                actual=self.collect(self.provider(path,size,num_workers=1 if size==72 else 4))
                for a,b in zip(actual,baseline):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_prp_order_and_checkpoint_after_conversion(self):
        legacy=self.write(72);small=self.write(7)
        expected=self.collect(self.provider(legacy,72,shuffle=True,num_workers=1,seed=17,epoch=2,prp_block_size=17,prp_buffer_blocks=3))
        for size in (7,8,9,25,73,144):
            actual=self.collect(self.provider(self.write(size),size,shuffle=True,num_workers=4,seed=17,epoch=2,prp_block_size=17,prp_buffer_blocks=3))
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        old=data.PrpBatchDataset('HalfKAv2',legacy,32)
        list(old);state=old.state_dict()
        new=data.PrpBatchDataset('HalfKAv2',small,32,record_bytes=7)
        new.load_state_dict(state)
        self.assertEqual(new.next_epoch,1)
        self.assertEqual(sum(len(b[7]) for b in new),137)
        del state['record_bytes']
        old.load_state_dict(state) # pre-change checkpoints are still accepted

    def test_compact_sequential_cycles_and_partial_prp(self):
        small=self.write(7)
        provider=data.SparseBatchProvider('HalfKAv2',small,32,cyclic=True,record_bytes=7)
        try:
            ids=[int(x) for _ in range(6) for x in next(provider)[7].flatten().tolist()]
            self.assertEqual(ids,[i-100 for i in range(137)]+[i-100 for i in range(55)])
        finally:provider.close()
        provider=self.provider(small,7,shuffle=True)
        try:self.assertEqual([len(b[7]) for b in provider],[32,32,32,32,9])
        finally:provider.close()

    def test_dense_bitmap_matches_packed_features(self):
        # 32 occupied squares fit at 27 bytes through bitmap mode.
        position=((0,63),[(i+1,i%5,i%2) for i in range(30)],0,31990,1)
        compact=self.collect(self.provider(self.write(27,[position]),27))
        legacy=self.collect(self.provider(self.write(72,[position]),72))
        for a,b in zip(compact,legacy):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_malformed_width_records_and_native_errors(self):
        for bad in (6,145,7.5):
            with self.assertRaises(ValueError):data.training_data_record_count('unused.bin',bad)
        path=pathlib.Path(self.write(7))
        path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):data.training_data_record_count(str(path),7)
        raw=bytearray(frame(*self.positions[0],7));raw[6]|=64
        path.write_bytes(raw)
        provider=self.provider(str(path),7,shuffle=True)
        try:
            with self.assertRaisesRegex(RuntimeError,'padding'):next(provider)
        finally:provider.close()


if __name__=='__main__':unittest.main()
