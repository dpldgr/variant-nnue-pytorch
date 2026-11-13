"""Runtime schema isolation, malformed inputs and feature/model consistency."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
import features
import variant
import nnue_dataset as data
from generic_fixtures import PROFILES,write


class GenericTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def provider(self,name):
        cfg=variant.VariantConfig.from_dict(PROFILES[name]);path=self.root/(name+'.bin')
        write(path,PROFILES[name])
        return data.PrpSparseBatchProvider('HalfKAv2^',str(path),32,num_workers=4,config=cfg)

    def test_independent_simultaneous_contexts(self):
        names=list(PROFILES)
        providers=[self.provider(name) for name in names]
        try:
            for p in providers:p.native.close() # stream retains immutable native context
            for _ in range(5):
                for name,p in zip(names,providers):
                    try:b=next(p)
                    except StopIteration:continue
                    fs=features.get_feature_set_from_name('HalfKAv2^',p.config)
                    for ids in (b[2],b[4]):
                        live=ids[ids>=0]
                        self.assertTrue((live<fs.num_features).all())
                    self.assertEqual(b[2].shape[1],2*p.config.PIECE_COUNT)
        finally:
            for p in providers:p.close()

    def test_json_validation_and_derived_values(self):
        cfg=variant.default_config()
        self.assertEqual((cfg.SQUARES,cfg.PIECES,cfg.POCKETS),(64,12,0))
        self.assertEqual(variant.VariantConfig.from_dict(cfg.to_dict()),cfg)
        for key,val in [('FILES',True),('RANKS',0),('PIECE_TYPES',0),('KING_SQUARES',65),('USE_POCKETS',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                variant.VariantConfig.from_dict(dict(cfg.to_dict(),**{key:val}))
        path=self.root/'duplicate.json';path.write_text('{"FILES":8,"FILES":9}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):variant.load_config(path)
        with self.assertRaises(ValueError):variant.VariantConfig.from_dict(dict(cfg.to_dict(),PIECES=12))

    def test_legacy_configuration_conversion(self):
        from scripts.convert_variant_config import convert
        python_file = self.root / 'old_variant.py'
        header_file = self.root / 'old_variant.h'
        python_file.write_text('''RANKS = 8
FILES = 8
SQUARES = RANKS * FILES
KING_SQUARES = 64
PIECE_TYPES = 6
PIECES = 2 * PIECE_TYPES
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0
PIECE_VALUES = {1: 126, 2: 2538, 3: 1506, 4: 1166, 5: 1093}
''')
        header_file.write_text('#define PIECE_COUNT 32\n')
        expected = variant.load_config(Path(__file__).resolve().parents[1] / 'configs/metal-chess.json')
        self.assertEqual(convert(python_file, header_file), expected)
        python_file.write_text("__import__('os').getcwd()\n")
        with self.assertRaisesRegex(ValueError, 'assignments only'):
            convert(python_file, header_file)

    def test_saved_configuration_and_layout_isolation(self):
        import model
        small=variant.VariantConfig.from_dict(PROFILES['rectangle'])
        fs=features.get_feature_set_from_name('HalfKAv2^',small)
        net=model.NNUE(fs);saved={};net.on_save_checkpoint(saved)
        self.assertEqual(saved['variant_config'],small.to_dict())
        self.assertEqual(variant.resolve_config(saved=saved['variant_config']),small)
        other=features.get_feature_set_from_name('HalfKAv2^')
        self.assertNotEqual(other.num_features,fs.num_features)
        bad=model.NNUE(other)
        with self.assertRaisesRegex(ValueError,'configuration'):bad.on_load_checkpoint(saved)
        p=self.root/'wrong.json';p.write_text(json.dumps(PROFILES['chess']))
        with self.assertRaisesRegex(ValueError,'configuration'):variant.resolve_config(p,saved['variant_config'])

    def test_coordinates_larger_than_eight_bits(self):
        from test_record_bytes import frame
        for files,ranks,types in ((16,16,6),(36,36,6),(36,36,301)):
            cfg=variant.VariantConfig.from_dict(dict(variant.default_config().to_dict(),FILES=files,RANKS=ranks,KING_SQUARES=files*ranks,PIECE_TYPES=types))
            path=self.root/f'large-{files}.bin'
            path.write_bytes(frame((cfg.SQUARES-1,0),[(cfg.SQUARES//2,types-2,0)],0,31999,1,8,cells=cfg.SQUARES,types=types-1))
            p=data.PrpSparseBatchProvider('HalfKAv2^',str(path),4,record_bytes=8,config=cfg)
            try:
                b=next(p)
                for color,ids in enumerate((b[2][0],b[4][0])):
                    king=cfg.SQUARES-1 if color==0 else (ranks-1)*files
                    expected=[]
                    pieces=[(cfg.SQUARES-1,types-1,0),(0,types-1,1),(cfg.SQUARES//2,types-2,0)]
                    for sq,kind,owner in sorted(pieces):
                        oriented=sq if color==0 else (ranks-1-sq//files)*files+sq%files
                        plane=kind*2+(owner!=color)
                        if plane==2*types-1:plane-=1
                        expected.append(oriented+plane*cfg.SQUARES+king*(2*types-1)*cfg.SQUARES)
                    base=(2*types-1)*cfg.SQUARES*cfg.SQUARES
                    for sq,kind,owner in sorted(pieces):
                        oriented=sq if color==0 else (ranks-1-sq//files)*files+sq%files
                        expected.append(base+(kind*2+(owner!=color))*cfg.SQUARES+oriented)
                    self.assertEqual(ids[ids>=0].tolist(),expected)
            finally:p.close()

    def test_generic_dimension_errors_reach_python(self):
        huge=variant.VariantConfig.from_dict(dict(variant.default_config().to_dict(),FILES=255,RANKS=257,KING_SQUARES=65535,PIECE_TYPES=300))
        with self.assertRaisesRegex(RuntimeError,'int32'):
            data.PrpSparseBatchProvider('HalfKAv2','unused.bin',32,config=huge)

    def test_malformed_legacy_records_reach_python(self):
        path=self.root/'bad.bin';path.write_bytes(bytes(72))
        for cls in (data.SparseBatchProvider,data.PrpSparseBatchProvider):
            p=cls('HalfKAv2',str(path),16)
            try:
                with self.assertRaises(RuntimeError):next(p)
            finally:p.close()

    def test_buffered_records_keep_features_and_targets_together(self):
        for name, values in PROFILES.items():
            cfg = variant.VariantConfig.from_dict(values)
            path = self.root / (name+'.bin')
            write(path, values)
            for family in ('HalfKP','HalfKP^','HalfKA','HalfKA^','HalfKAv2','HalfKAv2^'):
                if name == 'no-reference' and family.startswith('HalfKP'):
                    continue
                results = []
                for mode in ('global','buffered'):
                    provider = data.PrpSparseBatchProvider(family, str(path), 32, num_workers=4,
                                                         config=cfg, prp_mode=mode,
                                                         prp_block_size=7, prp_buffer_blocks=3)
                    try:batches = list(provider)
                    finally:provider.close()
                    tensors = [torch.cat([b[i] for b in batches]) for i in range(10)]
                    order = tensors[7].flatten().argsort()
                    results.append([t[order] for t in tensors])
                for a,b in zip(*results):
                    torch.testing.assert_close(a,b,rtol=0,atol=0,msg=f'{name} {family}')
