import numpy as np

ref = np.load("test_data/reference_output.npy").flatten()
tiny = np.loadtxt("tinyonnx_output.txt")

abs_diff = np.abs(tiny - ref)
rel_diff = abs_diff / (np.abs(ref) + 1e-6)

print("Max abs diff:", np.max(abs_diff))
print("Max rel diff:", np.max(rel_diff))

if np.max(rel_diff) > 1e-2:
    raise SystemExit("❌ Output mismatch exceeds threshold.")
else:
    print("✅ Output matches expected results.")
