"""Independent, specification-based DTX encoder for decoder tests."""
import struct

def header(count, files=8, ranks=8, symbols=('K','P','Q','R','B','N'), variant='metal-chess', maximum=4, distance_bits=11):
    sb=max(1,(files*ranks-1).bit_length());tb=max(1,(len(symbols)-1).bit_length())
    bits=2*sb+(maximum-2)*(sb+tb+1)+1+distance_bits
    stride=(bits+7)//8
    name=variant.encode('utf-8');catalog=b''.join(struct.pack('<H',len(s.encode()))+s.encode() for s in symbols)
    length=(64+len(name)+len(catalog)+7)&~7
    raw=bytearray(length);raw[:8]=b'DTXTRN\r\n'
    struct.pack_into('<HHIIHHHHBBBBIIQ',raw,8,1,64,length,stride,files,ranks,len(symbols)-1,maximum,sb,tb,distance_bits,1,len(name),len(catalog),count)
    raw[52:56]=bytes((1,0,1,1));raw[64:64+len(name)]=name;raw[64+len(name):64+len(name)+len(catalog)]=catalog
    return bytes(raw),stride,sb,tb

def frame(kings,pieces,turn,distance, files=8,ranks=8,symbols=('K','P','Q','R','B','N'),maximum=4,distance_bits=11,**_):
    _,stride,sb,tb=header(0,files,ranks,symbols,maximum=maximum,distance_bits=distance_bits)
    value=0;at=0
    def put(v,n):
        nonlocal value,at
        value|=(v&((1<<n)-1))<<at;at+=n
    put(kings[0],sb);put(kings[1],sb)
    for i in range(maximum-2):
        sq,kind,color=pieces[i] if i<len(pieces) else (0,-1,0)
        put(sq,sb);put(kind+1,tb);put(color,1)
    put(turn,1);put(distance,distance_bits)
    return value.to_bytes(stride,'little')

def labels(distance,bits=11):
    if distance==-(1<<(bits-1)):return 0,0
    result=1 if distance>0 else -1
    return result*(32000-abs(distance)),result

def write(path,rows,**schema):
    head,_,_,_=header(len(rows),**schema)
    path.write_bytes(head+b''.join(frame(*r,**schema) for r in rows))
    return str(path)
