import numpy as np

def test_val(val_float):
    f16 = np.float16(val_float)
    u16 = f16.view(np.uint16)
    print(f"Float: {val_float} -> Float16: {f16} -> Uint16 bits: {u16} ({hex(u16)})")

def test_bits(val_uint):
    u16 = np.uint16(val_uint)
    f16 = u16.view(np.float16)
    f32 = f16.astype(np.float32)
    print(f"Uint16: {val_uint} ({hex(val_uint)}) -> Float16: {f16} -> Float32: {f32}")

print("--- Check User Report ---")
# User says Max Depth is 65280.0
test_val(65280.0)

print("\n--- Check 0xFF00 ---")
test_bits(0xFF00)

print("\n--- Check Infinity ---")
test_val(np.inf)

print("\n--- Check 1.0 ---")
test_val(1.0)
