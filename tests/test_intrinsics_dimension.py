from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
import torch

if __name__ == "__main__":
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