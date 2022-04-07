"""
Tensorized (and parallelizable) pytorch implementation of the algorithm for intrinsic dimension estimation :
    * Maximum Likelihood Estimation appoach
    * TWO-NN 
@author: Pascal Tikeng
@date : 28-03-2022
"""

import torch
import numpy as np
from typing import List, Optional, Union, Tuple

@torch.no_grad()
def pairwise_distances(mat : torch.Tensor) -> torch.Tensor:
    """
    Computes the distance between pairs of elements of a matrix in a tensorized way : (x - y)^2 = x^2 - 2*x*y + y^2
    https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7

    Parameters:
        * mat : torch.Tensor(n, dim), 2d data matrix, samples on rows and features on columns.

    Returns:
        * d : torch.Tensor(n, n), d[i][j]^2 = (mat[i] - mat[j])^2 =  mat[i]^2 - 2*mat[i]*mat[j] + mat[j]^2
    """
    # get the product x * y with y = x.t()
    r = torch.mm(mat, mat.t()) # (n, n)
    # get the diagonal elements
    diag = r.diag().unsqueeze(0).expand_as(r) # (n, n)
    # compute the distance matrix
    distances = diag + diag.t() - 2*r # (n, n)
    return distances.sqrt() # (n, n)


@torch.no_grad() 
def mle_id(data : torch.Tensor, k, averaging_of_inverses : Optional[bool] = True) -> float:
    """
    Calculates intrinsic dimension of the provided data points with the Maximum Likelihood Estimation appoach.
    References: 
      [1] Elizaveta Levina and Peter J Bickel. Maximum likelihood estimation of intrinsic dimension. 
        In Advances in neural information processing systems, pp. 777–784, 2005.
        https://www.stat.berkeley.edu/~bickel/mldim.pdf
      [2] David J.C. MacKay and Zoubin Ghahramani. Comments on ‘maximum likelihood estimation of intrinsic dimension’ 
        by e. levina and p. bickel (2004), 2005. http://www.inference.org.uk/mackay/dimension/
      [3] THE INTRINSIC DIMENSION OF IMAGES AND ITS IMPACT ON LEARNING, Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein
        https://openreview.net/pdf?id=XJk19XzGq2J

    Parameters:
        * data : torch.Tensor(n, dim), 2d data matrix, samples on rows and features on columns.
        * k : int, The number of neighbors to consider (k < N)
        * averaging_of_inverses : bool (default=True), Use the mean ([1]) or the inverse of the mean of the inverses ([2])
                                 [2] is by far more accurate than [1], so we leave it as the default choice.

    Returns:
        * d : float, intrinsic dimension of the dataset according to the Maximum Likelihood Estimation appoach.

    Maths (paste in .tex file):
        * One of the main approaches to intrinsic dimension estimation is to examine a neighborhood around each point in the dataset, 
        and compute the Euclidean distance to the $k^{th}$ nearest neighbor. Assuming that density is constant within small neighborhoods, 
        the Maximum Likelihood Estimation (MLE) of [1] uses a Poisson process to model the number of points found by random sampling within 
        a given radius around each sample point. By relating the rate of this process to the surface area of the sphere, the likelihood equations 
        yield an estimate of the ID at a given point $x$ as: 
            $$\hat{m}_k(x) = \bigg[ \frac{1}{k-1} \sum_{j=1}^{k-1} log \frac{T_k(x)}{T_j(x)} \bigg]^{-1}$$
        where $T_j(x)$ is the Euclidean ($l_2$) distance from $x$ to its $j^{th}$ nearest neighbor. [1] propose to average the local estimates at 
        each point to obtain a global estimate ($n$ is the number of sample) 
            $$\hat{m}_k = \frac{1}{n} \sum_{i=1}^{n} \hat{m}_k(x_i)$$ 
        [2] suggestion a correction based on averaging of inverses 
            $$\hat{m}_k = \bigg[ \frac{1}{n} \sum_{i=1}^{n} \hat{m}_k(x_i)^{-1} \bigg]^{-1} = \bigg[ \frac{1}{n(k-1)} \sum_{i=1}^{n} \sum_{j=1}^{k-1} log \frac{T_k(x_i)}{T_j(x_i)} \bigg]^{-1}$$ 
    """
    N = data.size(0)
    assert k < N
    pw_dist = pairwise_distances(data) # (n, n)
    pw_dist = pw_dist.flatten()[1:].view(N-1, N+1)[:,:-1].reshape(N, N-1) # (n, n-1), remove the diagonal (0...)
    T = pw_dist.topk(k=k,dim=1,largest=False,sorted=True).values # (n, k)
    m_k_x = (T[:,-1].unsqueeze(-1) / T[:,:-1]).log().mean(dim=1)**-1 # (n,)
    if averaging_of_inverses : m_k = (m_k_x**-1).mean()**-1
    else : m_k = m_k_x.mean()
    return m_k.item()

@torch.no_grad()
def twonn_pytorch(data : torch.Tensor, return_xy : Optional[bool] = False) -> Union[float, Tuple[float, torch.Tensor, torch.Tensor]]:
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    References: E. Facco, M. d’Errico, A. Rodriguez & A. Laio 
                Estimating the intrinsic dimension of datasets by a minimal neighborhood information 
                https://doi.org/10.1038/s41598-017-11873-y

    Parameters:
        * data : torch.Tensor(n, dim), 2d data matrix, samples on rows and features on columns.
        * return_xy : bool (default=False), whether to return also the coordinate vectors used for the linear fit.

    Returns:
      * d : int, intrinsic dimension of the dataset according to TWO-NN.
      * x : 1d Tensor (optional), array with the log(μ) values.
      * y : 1d Tensor (optional), array with the −log(1−F^{emp}(μ_i))  values.
    """
    # 1. Compute the pairwise distances for each point in the dataset i = 1, …, N.
    #pw_dist = torch.cdist(x1=data, x2=data, p=2.0) # (n, n), work, but 'the cost'
    pw_dist = pairwise_distances(data) # (n, n)

    # 2. For each point i find the two shortest distances r1 and r2.
    #r1_r2 = pw_dist.topk(k=3,dim=1,largest=False,sorted=True).values[...,1:] # (n, 2), fast, but ...
    N = data.size(0)
    pw_dist = pw_dist.flatten()[1:].view(N-1, N+1)[:,:-1].reshape(N, N-1) # (n, n-1), remove the diagonal (0...)
    r1_r2 = pw_dist.topk(k=2,dim=1,largest=False,sorted=True).values # (n, 2)
    r1, r2 = r1_r2.unbind(dim=1) # (n,), (n,)
    # clean data : remove the elements for which r1 = 0 (note that r1!=0  ==> r2!= 0 because 0 <= r1 <= r2)
    """
    There will be a difference in result between this and the implementation of Francesco Mottes (_numpy) because I remove the examples for 
    which r1 = 0, while in the implementation of Francesco Mottes, r1 is always the distance between the considered element and its nearest 
    one located at a non-zero distance from it (i.e. he looks for the first non-zero r1, while we consider it first even if it is zero before 
    removing it afterwards)
    """
    non_zeros = r1 > 0
    r1 = r1[non_zeros]
    r2 = r2[non_zeros]
  
    # 3. For each point i compute $µ_i = r_1/r_2$
    mu = r2/r1 # (n,)

    #4. Compute the empirical cumulate $F^{emp}(μ)$ by sorting the values of $μ$ in an ascending order through a permutation $σ$, then define $F^{emp}(μ_{σ(i)}) = i / N$
    _, sigma = torch.sort(mu) # permutation function (indices) $σ(i)$
    N = mu.size(0)
    Femp = torch.arange(1, N+1, device=data.device) / N # i/N
    Femp[sigma] = Femp.clone() # $F^{emp}(μ_{σ(i)}) = i / N$

    # 5. Fit the points of the plane given by coordinates $\{(log(μ_i), −log(1−F^{emp}(μ_i)))|i=1,..., N\}$ with a straight line passing through the origin.
    x = mu.log() # (n,)
    y = 1-Femp # (n,)
    non_zeros = y > 0 # avoid having log(0)
    x = x[non_zeros] # (n',)
    y = y[non_zeros] # (n',)
    y = -y.log() # (n',)
    # Fit line through origin to get the dimension : Computes the vector z that approximately solves the equation x @ z = y
    #z = torch.lstsq(input = y, A = torch.vstack([x, torch.zeros(x.size(0))]).T).solution # (n', 1)
    if False :
        """
        UserWarning: torch.lstsq is deprecated in favor of torch.linalg.lstsq and will be removed in a future PyTorch release.
        torch.linalg.lstsq has reversed arguments and does not return the QR decomposition in the returned tuple (although it returns 
        other information about the problem). To get the qr decomposition consider using torch.linalg.qr.
        The returned solution in torch.lstsq stored the residuals of the solution in the last m - n columns of the returned value whenever m > n. 
        In torch.linalg.lstsq, the residuals in the field 'residuals' of the returned named tuple.
        The unpacking of the solution, as in 
            ```X, _ = torch.lstsq(B, A).solution[:A.size(1)]```
        should be replaced with
            ```X = torch.linalg.lstsq(A, B).solution``` (Triggered internally at  ../aten/src/ATen/native/BatchLinearAlgebra.cpp:3668.)
        """
        z = torch.lstsq(input = y, A = x.unsqueeze(-1)).solution # (n', 1)
        d = z[0][0].item()
    else :
        # Confusing documentation: https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
        #z = torch.linalg.lstsq(A = x.unsqueeze(-1), B=y, rcond=None).solution
        z = torch.linalg.lstsq(input = x.unsqueeze(-1), b=y, rcond=None).solution
        d = z[0].item()
    
    if return_xy:
        return d, x, y
    else: 
        return d

def twonn_numpy(data : np.array, return_xy : Optional[bool] = False) -> Union[float, Tuple[float, np.array, np.array]]:
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    From Francesco Mottes, https://github.com/fmottes/TWO-NN/blob/master/TwoNN/twonn_dimension.py

    -----------
    Parameters:
    
    data : 2d array-like
        2d data matrix. Samples on rows and features on columns.
    return_xy : bool (default=False)
        Whether to return also the coordinate vectors used for the linear fit.
        
    -----------
    Returns:
    
    d : int
        Intrinsic dimension of the dataset according to TWO-NN.
    x : 1d array (optional)
        Array with the log(mu) values.
    y : 1d array (optional)
        Array with the -log(F(mu_{sigma(i)})) values.
        
    -----------
    References:
    
    [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information (https://doi.org/10.1038/s41598-017-11873-y)
    """
    N = len(data)
    #mu = r2/r1 for each data point
    mu = []
    for i,x in enumerate(data):
        dist = np.sort(np.sqrt(np.sum((x-data)**2, axis=1)))
        r1, r2 = dist[dist>0][:2]
        mu.append((i+1,r2/r1))
        
    #permutation function
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))
    mu = dict(mu)
    #cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu: F_i[sigma_i[i]] = i/N

    #fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])
    #avoid having log(0)
    x = x[y>0]
    y = y[y>0]
    y = -1*np.log(y)

    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
        
    if return_xy:
        return d, x, y
    else: 
        return d
