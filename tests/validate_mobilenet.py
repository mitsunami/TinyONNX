import numpy as np

ref = np.load("test_data/reference_output.npy").flatten()
tiny = np.loadtxt("tinyonnx_output.txt")

abs_diff = np.abs(ref - tiny)
max_abs_diff = abs_diff.max()
mean_abs_diff = abs_diff.mean()
rms_diff = np.sqrt((abs_diff ** 2).mean())

rel_diff = abs_diff / (np.abs(ref) + 1e-8)
max_rel_diff = rel_diff.max()

print(f"Max absolute difference: {max_abs_diff:.6f}")
print(f"Mean absolute difference: {mean_abs_diff:.6f}")
print(f"RMS difference: {rms_diff:.6f}")
print(f"Max relative difference: {max_rel_diff:.6f}")

# Recommended clear thresholds:
max_abs_threshold = 1e-4
mean_abs_threshold = 1e-5

if max_abs_diff < max_abs_threshold and mean_abs_diff < mean_abs_threshold:
    print("✅ Output matches expected results.")
else:
    raise SystemExit("❌ Output mismatch exceeds threshold.")
