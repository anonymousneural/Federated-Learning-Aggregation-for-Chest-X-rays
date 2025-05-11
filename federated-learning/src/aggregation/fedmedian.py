def federated_median(weights_list):
    median_weights = []
    for layer_weights in zip(*weights_list):
        median_weights.append(np.median(layer_weights, axis=0))
    return median_weights