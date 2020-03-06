import numpy as np;
import math;
import matplotlib.pyplot as plt

u =np.array([3,4]);
v =np.array([30,40]);

print(np.linalg.norm(u));
print(np.linalg.norm(v));
# direction
w=u/np.linalg.norm(u);
z=v/np.linalg.norm(v);
print(np.linalg.norm(u));
print(u/np.linalg.norm(u));
print(v/np.linalg.norm(v));
print('direction');
print(w);
print(z);
print(np.linalg.norm(w));
print(np.linalg.norm(z));

def geomentricDotProduct(x,y,theta):
    x_norm=np.linalg.norm(x);
    y_norm=np.linalg.norm(y);
    return x_norm * y_norm*math.cos(math.radians(theta));

print(geomentricDotProduct([3,5],[8,2],60));
print(geomentricDotProduct([3,5],[8,2],10));
print(geomentricDotProduct([3,5],[8,2],.1));
print(geomentricDotProduct([3,5],[8,2],0));
print(geomentricDotProduct([3,5],[8,2],45));
print(geomentricDotProduct([3,5],[8,2],90));

def dot_product(X,y):
    result=0;
    for i in range(len(X)):
        result=result+  X[i]*y[i];
        # print('result');
        # print(result);
    return(result);
X=[3,5];
y=[8,2];

print(dot_product(X,y));
# x_temp=np.array([0,1,2,3]);
# y_temp=np.array([-1,0.2,0.9,2.1]);
x_temp=np.array([3,5]);
y_temp=np.array([8,2]);
A=np.vstack([x_temp,np.ones(len(x_temp))]).T;
print(A);
m,c = np.linalg.lstsq(A,y_temp)[0];
print(m);
plt.plot(x_temp,m*x_temp + c ,'b',label='line')
plt.legend();
plt.show();