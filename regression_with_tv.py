import torch

def solve3(self,normalized_augmentations,target_id,y_map):
    # Assuming X is your input tensor with shape (100, 512, 14, 14)
    # Assuming y is your target tensor with shape (100,1, 14, 14)

    # Reshape X to have dimensions (100, 512, 196)
    X = X.view(X.size(0), X.size(1), -1)

    # Reshape y to have dimensions (100, 196)
    y = y.view(y.size(0), -1)

    # Flatten X to have dimensions (100, 512*196)
    X = X.view(X.size(0), -1)

    # Compute the closed form solution
    X_transpose = torch.transpose(X, 0, 1)
    X_transpose_X = torch.matmul(X_transpose, X)
    X_transpose_y = torch.matmul(X_transpose, y)

    # Compute the smoothness constraint matrix A
    smoothness_matrix = torch.eye(y.size(1))  # Identity matrix as a starting point
    smoothness_matrix -= torch.eye(y.size(1), k=1)  # Subtract the upper diagonal
    smoothness_matrix -= torch.eye(y.size(1), k=-1)  # Subtract the lower diagonal

    # Adjust the regularization parameter lambda according to your needs
    lambda_value = 0.1

    # Add the regularization term to the X_transpose_X matrix
    X_transpose_X += lambda_value * torch.matmul(X_transpose, smoothness_matrix)  # Add smoothness constraint

    # Compute the closed form solution with regularization
    weights_bias = torch.matmul(torch.inverse(X_transpose_X), X_transpose_y)

    # Extract the weights and bias
    w = weights_bias[:-1]  # Exclude the last element (bias)
    b = weights_bias[-1]   # The last element is the bias term

    print("Weights (w):", w.shape)
    print("Bias (b):", b)
