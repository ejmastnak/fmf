import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Seed value
seed_value= 1000

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set random seed
random.seed(seed_value)

# 3. Set numpy seed
np.random.seed(seed_value)

# Number of sample points
N=10000

#gauss
random1 = np.random.random_sample(N) # generate N random samples
random2=np.random.random_sample(N)   

x=random1
y=random2

#plot  histogram!
plt.figure(figsize=(4,4))

theta = np.arange(0, np.pi / 2, 0.01) # sample theta from 0 to pi/2
plt.plot(np.cos(theta),np.sin(theta), color="black") # plots circle boundary
boundary = x*x + y*y # used to determine if a sample is insed or outside circle
all_points=np.ones(N)

# Determine if samples hit or missed inside of circle
hits = np.ma.masked_where(boundary < 1., all_points) # inside; x^2 + y^2 < 1
misses = np.ma.masked_where(boundary >= 1., all_points) # outside

# Evaluate integration
total_hits=np.sum(hits.mask.astype(int))
pi_estimate=total_hits/N*4.
print("Estimate: {}, pi = {}, Relative Error {}".format(pi_estimate,np.pi,1. - pi_estimate/np.pi))

plt.scatter(x, y, s=hits, marker='^', c='C0')
plt.scatter(x, y, s=misses, marker='o', c='C1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Monte-Carlo Integration')
plt.show()
#plt.savefig('dist.png')
