"""Independent exhaustive finite FP16-scale / signed-INT4 two-BF16 identity."""
import struct

def bf16_rne(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]

count = 0
for bits in range(65536):
    if bits & 0x7c00 == 0x7c00:
        continue
    scale = struct.unpack('<e', struct.pack('<H', bits))[0]
    for code in range(-8, 8):
        represented = code * scale
        high = bf16_rne(represented)
        low = bf16_rne(represented - high)
        assert high + low == represented, (bits, code, represented, high, low)
        count += 1
print(f'PASS exact represented value decomposition: {count} finite scale/code pairs')
