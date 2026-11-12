"""Small independent wire fixtures for comparing generic and compiled loaders."""
import json
from pathlib import Path
from test_record_bytes import frame

DEFAULT=json.loads((Path(__file__).parents[1]/'variant.json').read_text())
PROFILES={}
for name,files,ranks,types,pieces,kings,pockets in (
    ('chess',8,8,6,32,64,False),('rectangle',4,3,6,12,12,False),
    ('shogi',9,9,14,40,81,True),('palace',9,10,7,32,9,False),
    ('no-reference',8,8,5,24,1,True),('wide',16,8,6,32,128,False)):
    cfg=dict(DEFAULT,FILES=files,RANKS=ranks,PIECE_TYPES=types,PIECE_COUNT=pieces,
             KING_SQUARES=kings,USE_POCKETS=pockets)
    cfg['PIECE_VALUES']={str(i):i*100 for i in range(1,types)}
    PROFILES[name]=cfg

PROFILES['metal']=json.loads((Path(__file__).parents[1]/'configs/metal-chess.json').read_text())


def rows(cfg,count=137):
    cells=cfg['FILES']*cfg['RANKS'];types=cfg['PIECE_TYPES'];reference=cfg['KING_SQUARES']>1
    for i in range(count):
        if not reference:kings=(cells,cells)
        elif cfg['KING_SQUARES']==9:
            a=3+i%3+(i//3%3)*cfg['FILES']
            b=cells-cfg['FILES']+(3+i//7%3)-(i//11%3)*cfg['FILES']
            kings=(a,b)
        else:
            a=i*13%cells;b=(i*7+3)%cells
            kings=(a,(b+1)%cells if a==b else b)
        used=set(kings);supports=[]
        for j in range(1+i%3):
            sq=(i*11+j*17)%cells
            while sq in used:sq=(sq+1)%cells
            used.add(sq);supports.append((sq,(i+j)%(types-int(reference)),(i+j)%2))
        hands={}
        if cfg['USE_POCKETS']:
            hands[(0,i%2)]=i%7
            hands[(types-int(reference)-1,(i+1)%2)]=i%3
        yield kings,supports,i%2,i-100,i%3-1,hands


def packed(cfg,row):
    kings,pieces,turn,score,result,hands=row
    word=0;at=0
    def put(value,n):
        nonlocal word,at
        word|=(value&((1<<n)-1))<<at;at+=n
    put(turn,1);put(kings[0],7);put(kings[1],7)
    board={s:(t,c) for s,t,c in pieces};reference=cfg['KING_SQUARES']>1
    for rank in range(cfg['RANKS']-1,-1,-1):
        for file in range(cfg['FILES']):
            sq=rank*cfg['FILES']+file
            if reference and sq in kings:continue
            if sq in board:
                t,c=board[sq];put(2*t+1,5);put(c,1)
            else:put(0,1)
    for color in range(2):
        for t in range(cfg['PIECE_TYPES']):put(hands.get((t,color),0),5)
    put(0,4);put(0,1);put(0,6);put(1,16);put(0,1)
    assert at<=512
    import struct
    return word.to_bytes(64,'little')+struct.pack('<hHHbB',score,0,1,result,0)


def write(path,cfg,size=72,count=137):
    output=[]
    for row in rows(cfg,count):
        if size==72:output.append(packed(cfg,row))
        else:
            if row[5]:raise ValueError('compact fixture cannot carry pockets')
            output.append(frame(*row[:5],size,cells=cfg['FILES']*cfg['RANKS'],types=cfg['PIECE_TYPES']-1))
    Path(path).write_bytes(b''.join(output))
