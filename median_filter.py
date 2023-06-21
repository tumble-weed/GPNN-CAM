import torch
def weighted_median_filter(input, k, sigma, padding=0):
    # Create the weighting filter using a Gaussian function
    device = input.device
    if sigma is not None:
        filter_ = torch.tensor([[torch.exp(-((i - k // 2) ** 2 + (j - k // 2) ** 2) / (2 * sigma ** 2)) for i in range(k)] for j in range(k)], dtype=torch.float32,device=device)
        filter_ /= filter_.sum()
    else:
        filter_ = torch.ones((k, k), dtype=torch.float32,device=device)

     

    # Pad the input tensor with replication padding
    input_padded = torch.nn.functional.pad(input, (padding, padding, padding, padding), mode='replicate')

    # Unfold the padded input tensor to create a tensor of shape (b, c, k, k, h, w)
    input_unfolded = input_padded.unfold(2, k, 1).unfold(3, k, 1).flatten(start_dim=-2,end_dim=-1)

    # Calculate the weighted median by sorting the values and applying the weights
    # along the last two dimensions (h and w) and taking the dot product with the filter
    input_unfolded = input_unfolded * filter_.flatten()[None,None,None,None]
    # sorted_values, _ = input_unfolded.sort(dim=-1)
    # sorted_values, _ = sorted_values.sort(dim=-2)
    # weighted_medians = (sorted_values * filter_).sum(dim=(-1, -2))
    sorted_values = input_unfolded.sort(dim=-1)[0]
    weighted_medians = sorted_values[...,(k*k)//2]
    #import pdb;pdb.set_trace()
    return weighted_medians
##############################################################################################