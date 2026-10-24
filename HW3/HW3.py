import numpy as np

t = 2.0**(-20)
b = np.array([4.0, 4.0])
B = np.column_stack([b, b + t * np.array([1.0, -1.0]), b + t * np.array([1.0, 1.0])])

norm_b = np.linalg.norm(b)
rel_b = np.linalg.norm(B[:, 1:] - b[:, None], axis=0) / norm_b

print(f"{'k':<3} | {'eps':<10} | {'kappa(A)':<12} | {'Ratio (-1)':<15} | {'Ratio (+1)':<15}")
print("-" * 65)

for k in [1, 10, 20, 30, 40]:
    eps = 2.0**(-k)
    A = np.array([[2.0 + eps, 2.0 - eps], 
                  [2.0 - eps, 2.0 + eps]])
    
    X = np.linalg.solve(A, B)
    
    rel_x = np.linalg.norm(X[:, 1:] - X[:, [0]], axis=0) / np.linalg.norm(X[:, 0])
    ratio1, ratio2 = rel_x / rel_b

    print(f"{k:<3} | {eps:<10.2e} | {2.0 / eps:<12.4e} | {ratio1:<15.4e} | {ratio2:<15.4e}")        