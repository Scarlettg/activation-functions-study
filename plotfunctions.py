from numpy import *
import math
import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(-4,4,10000)
fig, ax = plt.subplots()

# ax.plot(x, np.maximum(np.zeros(len(x)),x),label="ReLU",linewidth=3)


# lrelu=x.copy()
# for i in range(len(x)):
# 	if lrelu[i]<0:
# 		lrelu[i]=lrelu[i]*0.2
# ax.plot(x, lrelu,label="LReLU",linewidth=1)


# ax.plot(x, (1-np.exp(-2*x))/(1+np.exp(-2*x)),label="Tanh")




# new=x.copy()
# for i in range(len(x)):
# 	if new[i]<0:
# 		new[i]=new[i]*exp(new[i])
# ax.plot(x, new,label="New activation function",linewidth=3)

# ax.plot(x, x/(1+exp(-x)),label="Swish")

new=x.copy()
for i in range(len(x)):
	if new[i]<0:
		new[i]=(1+new[i])*exp(new[i])
	else:
		new[i]=1
ax.plot(x, new,label="New activation function",linewidth=1)

def sig(x):
	return 1/(1+exp(-x))
ax.plot(x, sig(x)+x*sig(x)*(1-sig(x)),label="Swish")





# elu=x.copy()
# for i in range(len(x)):
# 	if elu[i]<0:
# 		elu[i]=np.exp(elu[i])-1
# ax.plot(x, elu,label="ELU")

ax.axhline(y=0, color='k',alpha=0.5)
ax.axvline(x=0, color='k',alpha=0.5)

legend = ax.legend(loc='upper left',fontsize=14)
plt.xlabel("x",fontsize=15)
plt.ylabel("y",fontsize=15)

plt.ylim(ymax=1.5)
plt.ylim(ymin=-0.5)
plt.show()