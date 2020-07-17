from numpy import array
from numpy import diag;
from numpy import dot;
from scipy.linalg import svd;

A = array([[1,2,3],[4,5,6],[7,8,9]])
print(A)
U,s,VT = svd(A)
print(U)
print(s)
print(VT)

Sigma=diag(s)

print(Sigma)

B=U.dot(Sigma.dot(VT));
print(B);
