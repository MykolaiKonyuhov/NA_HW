import numpy as np

def myL(L, b):
    """
    Пряма підстановка (Section 6.2.1, стор. 140 у книзі):
    Розв'язує Ly = b, де L — нижньотрикутна матриця з 1 на діагоналі.
    """
    n = len(b)
    y = b.astype(np.float64).copy()
    for k in range(1, n):
        # Відповідає: y[k] = b[k] - sum(L[k, j] * y[j] for j in 1:k-1)
        y[k] = b[k] - np.sum(L[k, :k] * y[:k])
    return y


def myU(U, y):
    """
    Зворотна підстановка (Section 6.2.1, стор. 140 у книзі):
    Розв'язує Ux = y, де U — верхньотрикутна матриця.
    """
    n = len(y)
    x = y.astype(np.float64).copy()
    for k in range(n - 1, -1, -1):
        # Відповідає: x[k] = (y[k] - sum(U[k, j] * x[j] for j in k+1:n; init=0)) / U[k, k]
        x[k] = (y[k] - np.sum(U[k, k + 1:] * x[k + 1:])) / U[k, k]
    return x


def myLU(A0):
    """
    LU-розклад без вибору ведучого елемента (Section 6.2.1, стор. 143 у книзі):
    Повертає L, U такі, що A = L @ U.
    """
    n = A0.shape[0]
    A = A0.astype(np.float64).copy()
    for k in range(n - 1):
        if np.isclose(A[k, k], 0.0):
            raise ZeroDivisionError(f"Нульовий ведучий елемент у позиції ({k}, {k})")
        # Обчислення множників l^(k) нижче діагоналі
        A[k + 1:n, k] /= A[k, k]
        # Оновлення підматриці B^(k+1)
        A[k + 1:n, k + 1:n] -= np.outer(A[k + 1:n, k], A[k, k + 1:n])
        
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    return L, U


def myLUpivot(A0):
    """
    LU-розклад із частковим вибором ведучого елемента (Section 6.2.2, стор. 145 у книзі):
    Повертає L, U, p такі, що L @ U = A[p, :], тобто P @ A = L @ U.
    """
    n = A0.shape[0]
    p = np.arange(n)
    A = A0.astype(np.float64).copy()
    
    for k in range(n - 1):
        # Пошук максимального за модулем елемента у поточному стовпці від рядка k до n-1
        mptr = np.argmax(np.abs(A[p[k:n], k]))
        # Перестановка індексів рядків у векторі перестановок p
        p[k], p[k + mptr] = p[k + mptr], p[k]
        
        # Обчислення коефіцієнтів матриці L
        A[p[k + 1:n], k] /= A[p[k], k]
        # Оновлення залишку матриці (підматриці)
        A[p[k + 1:n], k + 1:n] -= np.outer(A[p[k + 1:n], k], A[p[k], k + 1:n])
        
    L = np.tril(A[p, :], -1) + np.eye(n)
    U = np.triu(A[p, :])
    return L, U, p


def solve_linear_system(A, b, pivot=True):
    """
    Об'єднання підпрограм для розв'язання Ax = b:
    Якщо pivot=True:  PA = LU  ==>  Ly = Pb,  Ux = y
    Якщо pivot=False: A = LU   ==>  Ly = b,   Ux = y
    """
    if pivot:
        L, U, p = myLUpivot(A)
        # Вектор b переставляється відповідно до вектора перестановок p
        y = myL(L, b[p])
        x = myU(U, y)
    else:
        L, U = myLU(A)
        y = myL(L, b)
        x = myU(U, y)
    return x