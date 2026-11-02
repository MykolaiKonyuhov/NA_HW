import numpy as np
import matplotlib.pyplot as plt
#1
#a)
a = 1/2 * 10**7 - 1

def f1_expanded(x):
    return x**3 - 6*a * x**2 + 12 * a**2 * x - 8 * a**3

def f1_factored(x):
    return (x - 2*a)**3

x0 = 10**7

print(f"Expanded version: {f1_expanded(x0)}")
print(f"Factored version: {f1_factored(x0)}")

#b)
def f2_before(x):
    return 1/(x-np.sqrt(x**2 - 1))

def f2_after(x):
    return x + np.sqrt(x**2 - 1)

print(f"Before transformation: {f2_before(x0)}")
print(f"After transformation: {f2_after(x0)}")

delta = 100
x_vals = np.linspace(x0 - delta, x0 + delta, 2000)

y_before = f2_before(x_vals)
y_after = f2_after(x_vals)

plt.figure(figsize=(10, 5))

plt.plot(x_vals, y_before, label='Before (1 / (x - sqrt(x² - 1)))', color='crimson', linewidth=1.5)
plt.plot(x_vals, y_after, label='After (x + sqrt(x² - 1))', color='navy', linestyle='--', linewidth=2)

plt.title(f"$f_2(x)$ Near $x_0 = 10^7$")
plt.xlabel("x")
plt.ylabel("$f_2(x)$")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.ticklabel_format(useOffset=True, style='plain')

plt.show()

#c)
x0 = 1e-7
def f3_before(x):
    return 1.0 - np.sqrt(1.0 - x**2)

def f3_after(x):
    return x**2 / (1.0 + np.sqrt(1.0 - x**2))

print(f"f3 before transformation at x0: {f3_before(x0)}")
print(f"f3 after transformation at x0:  {f3_after(x0)}")

x_vals_c = np.linspace(0.0, 2 * x0, 2000)

y3_before = f3_before(x_vals_c)
y3_after = f3_after(x_vals_c)

plt.figure(figsize=(10, 5))

plt.plot(x_vals_c, y3_before, label='Before: $1 - \\sqrt{1 - x^2}$', color='crimson', linewidth=1.5)

plt.plot(x_vals_c, y3_after, label='After: $x^2 / (1 + \\sqrt{1 - x^2})$', color='navy', linestyle='--', linewidth=2)

plt.title("$f_3(x)$ near $x_0 = 10^{-7}$")
plt.xlabel("x")
plt.ylabel("$f_3(x)$")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.show()

#d)
def f4_exp(x):
    return x ** 9 - 18 * x ** 8 + 144 * x ** 7 - 672 * x ** 6 + 2016 * x ** 5 - 4032 * x ** 4 + 5376 * x ** 3 - 4608 * x ** 2 + 2304 * x - 512

def f4_factored(x):
    return (x - 2) ** 9

x_vals_d = np.linspace(1.925, 2.075, 2000)

y4_first = f4_exp(x_vals_d)
y4_second = f4_factored(x_vals_d)

plt.figure(figsize=(10, 5))

plt.plot(x_vals_d, y4_first, label="Expanded: $x^9 - 18x^8 + \dots$", color="crimson", linewidth=1)
plt.plot(x_vals_d, y4_second, label="Factored: $(x - 2)^9$", color="navy", linewidth=2)

plt.title("$f_4(x)$ on $[1.925, 2.075]$")
plt.xlabel("x")
plt.ylabel("$f_4(x)$")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.show()

#e)
def f5_before(x):
    return 1 / (1 + 2 * x) - (1 - x) / (1 + x)

def f5_after(x):
    return 2 * x ** 2 / ((1 + 2 * x) * (1 + x))

x_vals_e = np.linspace(-5e-8, 5e-8, 2000)

y5_before = f5_before(x_vals_e)
y5_after = f5_after(x_vals_e)

plt.figure(figsize=(10, 5))

plt.plot(x_vals_e, y5_before, label="Before: $1/(1+2x) - (1-x)/(1+x)$", color="crimson", linewidth=1.2)
plt.plot(x_vals_e, y5_after, label="After: $2x^2 / ((1+2x)(1+x))$", color="navy", linestyle="--", linewidth=2)

plt.title("$f_5(x)$ on $[-5\\cdot 10^{-8}, 5\\cdot 10^{-8}]$")
plt.xlabel("x")
plt.ylabel("$f_5(x)$")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.show()

#2 
#a)
def constr_A(n):
    if n < 5:
        raise ValueError("n must be at least 5")
    
    main_diag = 3.0 * np.ones(n)
    upper_diag = 2.0 * np.ones(n - 1)
    lower_diag = 1.0 * np.ones(n - 1)
    
    A = np.diag(main_diag) + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)
    return A

#b)
n = 50
A = constr_A(n)

t = np.linspace(0, 1, n)
x_true = np.sin(2 * np.pi * t)

y = A @ x_true

plt.figure(figsize=(10, 4))
plt.plot(t, x_true, label="Original signal $x$")
plt.plot(t, y, label="Output $y = Ax$")
plt.legend()
plt.grid(True)
plt.show()

#c)
noise_levels = [1e-4, 1e-2, 0.1]

plt.figure(figsize=(12, 5))
plt.plot(t, x_true, label="True $x$", color="black", linewidth=2)

for sigma in noise_levels:
    epsilon = np.random.normal(0, sigma, n)
    y_noisy = y + epsilon
    x_reconstructed = np.linalg.solve(A, y_noisy)
    plt.plot(t, x_reconstructed, label=f"Reconstructed (noise $\\sigma = {sigma}$)")

plt.title("Reconstruction of signal in the presence of noise")
plt.xlabel("t")
plt.legend()
plt.grid(True)
plt.show()

#d)
n_range = range(5, 101)
norms_1 = []

for k in n_range:
    Ak = constr_A(k)
    Ak_inv = np.linalg.inv(Ak)
    norm_val = np.linalg.norm(Ak_inv, ord=1)
    norms_1.append(norm_val)

plt.figure(figsize=(10, 4))
plt.plot(list(n_range), norms_1, marker='.', color='purple')
plt.title("$\|A^{-1}\|_1$ depending on $n$")
plt.xlabel("$n$")
plt.ylabel("$\|A^{-1}\|_1$")
plt.grid(True)
plt.show()