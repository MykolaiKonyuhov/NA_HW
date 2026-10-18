import numpy as np
import matplotlib.pyplot as plt

#1

print("--- 1. Integer Overflow ---")

a_py = 2**63 - 1
b_py = -(2**63)

print("a + 1 =", a_py + 1)  
print("b - 1 =", b_py - 1)  

# Numpy int64 as in Julia
a_np = np.int64(2**63 - 1)
b_np = np.int64(-(2**63))

print("a + 1 =", a_np + np.int64(1))  
print("b - 1 =", b_np - np.int64(1))  

#2

# Absorption & Machine Epsilon
x = 2.0**(-52)  #eps
print("5.0 + x == 5.0:", (5.0 + x) == 5.0)  


#3

#Underflow & Subnormal numbers

print("2^(-1074) =", 2.0**(-1074))         
print("2^(-1075) =", 2.0**(-1075))         
print("2^(-1075) == 0.0:", 2.0**(-1075) == 0.0)  # True

#4

#Loss of Associativity

# f(x, y) = 4x^4 + 2y^2 - y^4
# g(x, y) = 4x^4 - y^4 + 2y^2


def f(x, y):
    return 4.0 * x**4 + 2.0 * y**2 - y**4

def g(x, y):
    return 4.0 * x**4 - y**4 + 2.0 * y**2

x_val, y_val = 13860.0, 19601.0
print("f(13860, 19601) =", f(x_val, y_val))
print("g(13860, 19601) =", g(x_val, y_val))

#5

# f(x) = (x - 2a)^3 

#Polynomial Cancellation

a = 4999999.0
x0 = 10.0**7

def f_poly(x):
    return x**3 - 6.0 * a * x**2 + 12.0 * (a**2) * x - 8.0 * (a**3)

def g_poly(x):
    return (x - 2.0 * a)**3

print("f(x0) =", f_poly(x0))  
print(" g(x0)  =", g_poly(x0))  

#6 

#Cancellation with Square Roots

x = 10.0**7
f_sqrt = 1.0 / (x - np.sqrt(x**2 - 1.0))
g_sqrt = x + np.sqrt(x**2 - 1.0)
print("f(x) - g(x) =", f_sqrt - g_sqrt)  

#7

#Ill-conditioned Linear Systems

A = np.array([[1.2969, 0.8648],
              [0.2161, 0.1441]])
b = np.array([0.8642, 0.1440])
x_exact = np.array([2.0, -2.0])
print("Ax - b:", np.linalg.norm(A @ x_exact - b))


delta_b = 1e-8 * np.array([-1.0, 1.0])
b_tilde = b + delta_b
x_tilde = np.array([0.9911, -0.4870])
print("Ax̃ - b̃:", np.linalg.norm(A @ x_tilde - b_tilde))
print("A:", np.linalg.cond(A))


#8

x_plot = np.linspace(1.925, 2.075, 1000)

def f_deg9(x):
    return (x**9 - 18.0*x**8 + 144.0*x**7 - 672.0*x**6 + 2016.0*x**5 
            - 4032.0*x**4 + 5376.0*x**3 - 4608.0*x**2 + 2304.0*x - 512.0)

def g_deg9(x):
    return (x - 2.0)**9

plt.figure(figsize=(9, 5))
plt.plot(x_plot, f_deg9(x_plot), label=r"$f(x)$ (expanded form, numerical noise)", color="red", lw=1)
plt.plot(x_plot, g_deg9(x_plot), label=r"$g(x) = (x - 2)^9$ (factored form)", color="blue", lw=2)
plt.title(r"Loss of Significance Near a Multiple Root of Order 9: $f(x)$ vs $g(x)$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()