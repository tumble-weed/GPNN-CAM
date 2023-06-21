import torch
import torch
if False:
    def solve_standard_regression(X,y,alpha=1000):
        # if True:
        # Add a column of ones to X for the bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1,device=X.device)), dim=1)

        # Remove redundant columns from X
        u, s, v = torch.svd(X)
        tol = s.max() * X.shape[1] * torch.finfo(s.dtype).eps
        mask = s > tol
        """
        X = torch.mm(u[:, mask], torch.diag(s[mask]).mm(v[:, mask].t()))

        # Calculate the OLS solution for W and b
        X_t = torch.transpose(X, 0, 1)  # Transpose of X
        XX_t_inv = torch.inverse(torch.matmul(X_t, X))  # Inverse of (X^T * X)
        """
        """
        Xt_X = torch.mm(v[ mask,:].t(), torch.diag(s[mask]**2).mm(v[ mask,:]))
        XX_t_inv = torch.inverse(Xt_X)
        W = torch.matmul(torch.matmul(XX_t_inv, X_t), y.unsqueeze(1))[:-1]  # W = (X^T * X)^-1 * X^T * y, excluding the last element (bias term)
        """
        u = u[:, mask]
        s = s[mask]
        v = v[:, mask]

        # calculate weights using OLS solution
        XtX_inv = torch.mm(v, torch.diag(1/s)).mm(v.t())
        X = torch.mm(u[:, mask], torch.diag(s[mask]).mm(v[:, mask].t()))
        Xt = X.T
        W,b = solve_lasso_regression_(XtX_inv, Xt,y, alpha)
        return W,b

    def solve_lasso_regression_(XtX_inv, Xt,y, alpha):
        # Compute the number of features
        num_features = XtX_inv.shape[0]

        Xty = torch.matmul(Xt, y.unsqueeze(1))

        # Initialize the coefficients with zeros
        W = torch.zeros(num_features, 1, device=XtX_inv.device)

        for _ in range(num_features):
            # Compute the correlation between features and the current residual
            correlations = torch.matmul(XtX_inv, W) - Xty

            # Find the feature with the highest positive correlation
            max_correlation_idx = torch.argmax(correlations)

            # Update the corresponding coefficient
            if correlations[max_correlation_idx] > alpha:
                W[max_correlation_idx] = (correlations[max_correlation_idx] - alpha) / XtX_inv[max_correlation_idx, max_correlation_idx]
            elif correlations[max_correlation_idx] < -alpha:
                W[max_correlation_idx] = (correlations[max_correlation_idx] + alpha) / XtX_inv[max_correlation_idx, max_correlation_idx]
            else:
                W[max_correlation_idx] = 0.0

        # Extract the bias term
        b = W[-1]
        W = W[:-1]
        W = W.squeeze()
        # Print the value of W and b
        import ipdb;ipdb.set_trace()
        print(f"W = {W}")
        print(f"b = {b.item()}")

        return W, b

elif False:
    def solve_standard_regression(X,y,device=None,alpha=None):
        # if True:
        # Add a column of ones to X for the bias term
        X = torch.cat((X, torch.ones(X.shape[0], 1,device=X.device)), dim=1)

        # Remove redundant columns from X
        u, s, v = torch.svd(X)
        tol = s.max() * X.shape[1] * torch.finfo(s.dtype).eps
        mask = s > tol
        """
        X = torch.mm(u[:, mask], torch.diag(s[mask]).mm(v[:, mask].t()))

        # Calculate the OLS solution for W and b
        X_t = torch.transpose(X, 0, 1)  # Transpose of X
        XX_t_inv = torch.inverse(torch.matmul(X_t, X))  # Inverse of (X^T * X)
        """
        """
        Xt_X = torch.mm(v[ mask,:].t(), torch.diag(s[mask]**2).mm(v[ mask,:]))
        XX_t_inv = torch.inverse(Xt_X)
        W = torch.matmul(torch.matmul(XX_t_inv, X_t), y.unsqueeze(1))[:-1]  # W = (X^T * X)^-1 * X^T * y, excluding the last element (bias term)
        """
        u = u[:, mask]
        s = s[mask]
        v = v[:, mask]

        # calculate weights using OLS solution
        XtX_inv = torch.mm(v, torch.diag(1/s)).mm(v.t())
        # W = XtX_inv.mm(X.t()).mm(y)
        W_and_b = XtX_inv.mm(X.t()).mm(y[:,None])
        W_and_b = W_and_b.squeeze()
        b = W_and_b[-1]  # Extract the bias term from the last element of W
        W = W_and_b[:-1]
        if False:
            # Print the value of W and b
            print(f"W = {W.squeeze()}")
            print(f"b = {b.item()}")
        return W,b
elif 'sklearn regression' and True:
    def solve_standard_regression(X,Y,device='cpu',alpha=0,sample_weight=None):
        # alpha=0.1
        from sklearn import linear_model
        if alpha != 0:
            print('alpha',alpha)
            # clf = linear_model.Lasso(alpha=alpha)
            clf = linear_model.Ridge(alpha=alpha)
            # import ipdb;ipdb.set_trace()
            # clf = linear_model.ElasticNet(alpha=alpha,l1_ratio=0.5)
        else:
            clf = linear_model.LinearRegression()
        # clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
        clf.fit(X.cpu().numpy(),Y.cpu().numpy(),sample_weight=sample_weight)
        w = clf.coef_
        b = clf.intercept_
        w,b = torch.tensor(w,device=device),torch.tensor(b,device=device)
        return w,b
elif "sklearn random forest" and False: 
    def solve_standard_regression(X,Y,device='cpu',alpha=0):
        # alpha=0.1
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X.cpu().numpy(),Y.cpu().numpy())
        w = regr
        b = None
        # w,b = torch.tensor(w,device=device),torch.tensor(b,device=device)
        return w,b    
elif 'ridge-copied from github' and False:
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Ridge:
        def __init__(self, alpha = 0, fit_intercept = True,):
            alpha = 0
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            
        def fit(self, X: torch.tensor, y: torch.tensor) -> None:
            
            device = X.device
            X = X.rename(None)
            y = y.rename(None).view(-1,1)
            assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
            if self.fit_intercept:
                X = torch.cat([torch.ones(X.shape[0], 1,device=device), X], dim = 1)
            # Solving X*w = y with Normal equations:
            # X^{T}*X*w = X^{T}*y 
            lhs = X.T @ X 
            rhs = X.T @ y
            # if rhs.ndim == 2:
            #     rhs= rhs.squeeze()
            if self.alpha == 0:
                self.w, residual,rank,singular_values = torch.linalg.lstsq(rhs, lhs)
            else:
                ridge = self.alpha*torch.eye(lhs.shape[0],device=device)
                self.w, residual,rank,singular_values = torch.linalg.lstsq(rhs, lhs + ridge)
            if self.w.ndim == 2:
                assert self.w.shape[0] == 1
                self.w = self.w.squeeze(0)
                # import ipdb;ipdb.set_trace()
        def predict(self, X: torch.tensor) -> None:
            X = X.rename(None)
            if self.fit_intercept:
                X = torch.cat([torch.ones(X.shape[0], 1,device=X.device), X], dim = 1)
            return X @ self.w
    def solve_standard_regression(X,y,device,alpha):
        reg = Ridge(alpha)
        reg.fit(X,y)
        return reg.w[:-1],reg.w[-1]
        
    
if __name__ == "__main__":
    ## demo
    X = torch.randn(100,3)
    y = torch.randn(100,1) # supports only single outputs

    model = Ridge(alpha = 1e-3, fit_intercept = True)
    model.fit(X,y)
    model.predict(X)
    
def solve_shared_regression(X,Y,lambda_ = 0.01):
    # Y = torch.log(prob)
    # Define the L1 regularization coefficient lambda
    # lambda_ = 0.01
    # lambda_ = 0.01
    # Flatten the Y tensor to shape (50, 1)
    Y = Y.reshape(-1, 1)

    # Compute the closed-form solution for w with L1 regularization
    X_2_sum = torch.sum(X ** 2, dim=(0, 2))
    X_Y_sum = torch.sum(X * Y[:, None, :], dim=(0, 2))
    X_sum = torch.sum(X, dim=(0, 2))
    Y_sum = torch.sum(Y, dim=0)
    n_samples = X.shape[0]
    w = torch.zeros_like(X[0, :, 0])
    for i in range(w.shape[0]):
        denominator = X_2_sum[i]
        numerator = X_Y_sum[i] - ((1 / n_samples) * X_sum[i] * Y_sum)
        if numerator > lambda_:
            w[i] = (numerator - lambda_) / denominator
        elif numerator < -lambda_:
            w[i] = (numerator + lambda_) / denominator
        else:
            w[i] = 0

    # Compute the bias b
    b = torch.mean(Y - torch.einsum('ijk,j->i', X, w))
    if False:
        # Print the weight vector w and the bias b
        print(w)
        print(b)
    # import ipdb;ipdb.set_trace()
    return w,b        
