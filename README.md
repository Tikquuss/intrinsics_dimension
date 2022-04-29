# Tensorized (and parallelizable) pytorch implementation of the algorithm for intrinsic dimension estimation :

## 1. Maximum Likelihood Estimation appoach

Calculates intrinsic dimension of the provided data points with the Maximum Likelihood Estimation appoach.

References: 

* [1] Elizaveta Levina and Peter J Bickel. Maximum likelihood estimation of intrinsic dimension. \
In Advances in neural information processing systems, pp. 777–784, 2005. \
https://www.stat.berkeley.edu/~bickel/mldim.pdf \
* [2] David J.C. MacKay and Zoubin Ghahramani. Comments on ‘maximum likelihood estimation of intrinsic dimension’ 
by e. levina and p. bickel (2004), 2005. http://www.inference.org.uk/mackay/dimension/
* [3] THE INTRINSIC DIMENSION OF IMAGES AND ITS IMPACT ON LEARNING, Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein \
https://openreview.net/pdf?id=XJk19XzGq2J


One of the main approaches to intrinsic dimension estimation is to examine a neighborhood around each point in the dataset, and compute the Euclidean distance to the $k^{th}$ nearest neighbor. Assuming that density is constant within small neighborhoods, the Maximum Likelihood Estimation (MLE) of [1] uses a Poisson process to model the number of points found by random sampling within a given radius around each sample point. By relating the rate of this process to the surface area of the sphere, the likelihood equations yield an estimate of the ID at a given point $x$ as: 
$$\hat{m}_k(x) = \bigg[ \frac{1}{k-1} \sum_{j=1}^{k-1} log \frac{T_k(x)}{T_j(x)} \bigg]^{-1}$$  
where $T_j(x)$ is the Euclidean ($l_2$) distance from $x$ to its $j^{th}$ nearest neighbor. [1] propose to average the local estimates at each point to obtain a global estimate ($n$ is the number of sample) 
$$\hat{m}_k = \frac{1}{n} \sum_{i=1}^{n} \hat{m}_k(x_i)$$ 
[2] suggestion a correction based on averaging of inverses 
$$\hat{m}_k = \bigg[ \frac{1}{n} \sum_{i=1}^{n} \hat{m}_k(x_i)^{-1} \bigg]^{-1} = \bigg[ \frac{1}{n(k-1)} \sum_{i=1}^{n} \sum_{j=1}^{k-1} log \frac{T_k(x_i)}{T_j(x_i)} \bigg]^{-1}$$ 

## 2. TWO-NN
Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.

References: 
* E. Facco, M. d’Errico, A. Rodriguez & A. Laio \
Estimating the intrinsic dimension of datasets by a minimal neighborhood information \
https://doi.org/10.1038/s41598-017-11873-y

2-NN algorithm :
1. Compute the pairwise distances for each point in the dataset $i = 1, …, N$.
2. For each point $i$ find the two shortest distances $r_1$ and $r_2$.
3. For each point $i$ compute $µ_i = \frac{r_2}{r_1}$
4. Compute the empirical cumulate $F^{emp}(μ)$ by sorting the values of $μ$ in an ascending order through a permutation $σ$, then define $F^{emp}(μ_{σ(i)}) = \frac{1}{N}$
5. Fit the points of the plane given by coordinates $\{(log(μ_i), −log(1−F^{emp}(μ_i))) \ | \ i=1,..., N\}$ with a straight line passing through the origin.

## 3. Installation
pip install intrinsics_dimension

## 4. Get started

```python

from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch

n, dim = 512, 1024
data = torch.randn(n, dim)
d1 = mle_id(data, k=2, averaging_of_inverses = False)
d2 = mle_id(data, k=2, averaging_of_inverses = True)
d3 = twonn_numpy(data.numpy(), return_xy=False)
d4 = twonn_pytorch(data, return_xy=False)
print(d1, d2, d3, d4)

data[1] = data[0] # make distance(data[1], data[0]) = 0
d1 = mle_id(data, k=2, averaging_of_inverses = False)
d2 = mle_id(data, k=2, averaging_of_inverses = True)
d3 = twonn_numpy(data.numpy(), return_xy=False)
d4 = twonn_pytorch(data, return_xy=False)
print(d1, d2, d3, d4)
```