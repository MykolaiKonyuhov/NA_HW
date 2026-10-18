#2
import numpy as np

def myL(L, b):
    y = b.astype(float).copy()
    for k in range(1, len(b)):
        y[k] -= L[k, :k] @ y[:k]
    return y

def myU(U, y):
    x = y.astype(float).copy()
    for k in reversed(range(len(y))):
        x[k] = (x[k] - U[k, k + 1:] @ x[k + 1:]) / U[k, k]
    return x

def myLU(A):
    A = A.astype(float).copy()
    n = len(A)
    for k in range(n - 1):
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
    return np.tril(A, -1) + np.eye(n), np.triu(A)

def myLUpivot(A):
    A = A.astype(float).copy()
    n = len(A)
    p = np.arange(n)
    for k in range(n - 1):
        m = k + np.argmax(np.abs(A[p[k:], k]))
        p[k], p[m] = p[m], p[k]
        A[p[k + 1:], k] /= A[p[k], k]
        A[p[k + 1:], k + 1:] -= np.outer(A[p[k + 1:], k], A[p[k], k + 1:])
    return np.tril(A[p], -1) + np.eye(n), np.triu(A[p]), p

def solve_system(A, b, pivot=True):
    if pivot:
        L, U, p = myLUpivot(A)
        return myU(U, myL(L, b[p]))
    L, U = myLU(A)
    return myU(U, myL(L, b))

# Example 6.2.4 / 6.1.8
A = np.array([[2., 4., -2.], [4., 9., -3.], [-2., -3., 7.]])
b = np.array([1., 3., 2.])
print("x:", solve_system(A, b))

print("Task 3 starts from here ")
#3

tests = [
    (np.float32, 25, "Float32"),
    (np.float64, 55, "Float64"),
    (np.longdouble, 67, "LongDouble")
]

for dtype, p, name in tests:
    eps = dtype(2.0) ** (-p)
    A = np.array([[eps, 1.0], [1.0, 1.0]], dtype=dtype)
    b = np.array([1.0, 2.0], dtype=dtype)
    x_exact = np.array([1.0 / (1.0 - eps), (1.0 - 2.0 * eps) / (1.0 - eps)], dtype=dtype)

    # Without pivoting (Example 6.4.3)
    L_no, U_no = myLU(A)
    x_no = myU(U_no, myL(L_no, b))

    # With pivoting (Example 6.4.4)
    L_p, U_p, perm = myLUpivot(A)
    x_p = myU(U_p, myL(L_p, b[perm]))

    print(f"=== {name} (eps = 2^-{p}) ===")
    print(f"Exact solution   : {x_exact}")
    print(f"myLU      (error): {np.linalg.norm(x_no - x_exact):.2e}")
    print(f"myLUpivot (error): {np.linalg.norm(x_p - x_exact):.2e}\n")