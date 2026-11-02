"""List or extract files from an id Tech 5 (Wolfenstein: The New Order) chunkN.index/.resources pair.

    python resources.py <chunk.index> [--grep SUBSTR] [--out DIR]

Index layout: 0x20 bytes of header, u32 BE ?, u32 BE entry count; then per entry u32 BE id,
three strings (u32 LE length + bytes: type, name, path), u32 BE offset, u32 BE size, u32 BE
stored size, 9 bytes of flags. A stored size below the size means zlib.
"""
import struct, sys, zlib, os, argparse

def entries(index):
    b = open(index, 'rb').read()
    count = struct.unpack_from('>I', b, 0x24)[0]
    p = 0x28
    for _ in range(count):
        if p + 4 > len(b):
            return
        ident = struct.unpack_from('>I', b, p)[0]; p += 4
        strs = []
        for _ in range(3):
            n = struct.unpack_from('<I', b, p)[0]; p += 4
            if n > 4096:
                print(f'index: entry layout lost at {p - 4:#x}, stopping', file=sys.stderr)
                return
            strs.append(b[p:p + n].decode('latin1')); p += n
        offset, size, stored = struct.unpack_from('>III', b, p); p += 12
        flags = b[p:p + 9]; p += 9
        yield dict(id=ident, type=strs[0], name=strs[1], path=strs[2], offset=offset, size=size, stored=stored, flags=flags)

def read(resources, e):
    with open(resources, 'rb') as f:
        f.seek(e['offset'])
        data = f.read(e['stored'])
    if e['stored'] < e['size']:
        # Raw deflate, flushed without a final block (zlib.decompress would call it truncated).
        out = zlib.decompressobj(-15).decompress(data)
        if len(out) != e['size']:
            raise ValueError(f"{e['path']}: inflated {len(out)} of {e['size']} bytes")
        return out
    return data

def main():
    a = argparse.ArgumentParser()
    a.add_argument('index'); a.add_argument('--grep', default=''); a.add_argument('--out')
    o = a.parse_args()
    res = o.index[:-len('.index')] + '.resources'
    n = 0
    for e in entries(o.index):
        if o.grep not in e['path']:
            continue
        n += 1
        if not o.out:
            print(e['id'], e['type'], e['path'], e['offset'], e['size'], e['stored'], e['flags'].hex())
            continue
        dest = os.path.join(o.out, e['path'].replace('/', os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, 'wb').write(read(res, e))
    print(n, 'entries', file=sys.stderr)

if __name__ == '__main__':
    main()
