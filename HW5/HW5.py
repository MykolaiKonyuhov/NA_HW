#1
#b
import math 

def myNaivePQ(p, q):
    d = math.sqrt(p ** 2 - q)
    x1 = p + d
    x2 = p - d
    result = (x1, x2)
    return result

#c

def myPQ(p, q):
    d = math.sqrt(p ** 2 - q)
    if p >= 0:
        x1 = p + d 
        x2 = q / x1
    else:
        x1 = p - d
        x2 = q / x1
    result = (x1, x2)
    return result

p = 1e7
q = 1
print(f"Result of myNaivePQ with p = {p} and q = {q} is {myNaivePQ(p, q)}")
print(f"And the resuult for myPQ is {myPQ(p, q)}")