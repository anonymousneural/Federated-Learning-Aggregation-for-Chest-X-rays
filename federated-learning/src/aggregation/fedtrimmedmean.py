def federated_trimmed_mean(weights_list, beta=None):
    if beta is None:
        beta = CONFIG['trimmed_mean_beta']
    
    trimmed_weights = []
    
    for layer_weights in zip(*weights_list):
        layer_weights_array = np.array(layer_weights)
        
        num_clients = len(weights_list)
        k = int(np.floor(beta * num_clients))
        
        if k > 0 and num_clients > 2 * k:
            trimmed_layer_weights = np.partition(layer_weights_array, k, axis=0)[k:num_clients - k]
            trimmed_weights.append(np.mean(trimmed_layer_weights, axis=0))
        else:
            trimmed_weights.append(np.mean(layer_weights_array, axis=0))
    
    return trimmed_weights