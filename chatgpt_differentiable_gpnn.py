"""
i have a function called get nearest neighbors that takes as input patches arranged in a tensor of shape (batch_size,channels,n_patch_rows,n_patch_cols,D), faiss index `index`, the number of nearest neighbors required `k` .
it returns the indexes of the k nearest neighbors of each patch arranged as (batch_size,channels,n_patch_rows,n_patch_cols,k)
"""
