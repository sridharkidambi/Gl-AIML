import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
# for discrete values the binomial and poisson is used 

n=7
p=.6
k=np.arange(0,8)
print(k)
binomial=stats.binom.pmf(k,n,p)
print(binomial)
print(sum(binomial))


# plt.plot(k,binomial)
# plt.title("binomial distribution")
# plt.xlabel("customer distribution-no of success")
# plt.ylabel("binomial probability success ")
# plt.show()

# rate=3 
# k1=np.arange(0,20)
# print(k1)
# poisson=stats.poisson.pmf(k1,rate)
# print(poisson[4])
# plt.plot(k1,poisson)
# plt.title("poisson distribution")
# plt.xlabel("customer rate -number  of customers/min")
# plt.ylabel("poisson probability success rate ")
# plt.show()

# for continuous values:

z_value= (.280-.295)/.025
print(z_value)

# cummulative dsitribution function
print(stats.norm.cdf(z_value))
print(stats.norm.cdf(.280,loc=.295,scale=.025))
print(1-stats.norm.cdf(.350,loc=.295,scale=.025))


lessthan =stats.norm.cdf(.260,loc=.295,scale=.025)
grtthan =1-stats.norm.cdf(.340,loc=.295,scale=.025)
print(1-grtthan-lessthan)
print(stats.norm.cdf(.340,loc=.295,scale=.025)-stats.norm.cdf(.260,loc=.295,scale=.025))

print(stats.norm.cdf(1)- stats.norm.cdf(-1))
print(stats.norm.cdf(2)- stats.norm.cdf(-2))
print(stats.norm.cdf(3)- stats.norm.cdf(-3))
