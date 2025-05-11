def federated_avg(weights_list):
    avg_weights = []
    for layer_weights in zip(*weights_list):
        avg_weights.append(np.mean(layer_weights, axis=0))
    return avg_weights