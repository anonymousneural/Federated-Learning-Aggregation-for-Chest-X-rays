def federated_prox(weights_list, global_weights, mu):
    avg_weights = []
    for layer_weights in zip(*weights_list):
        avg_weights.append(np.mean(layer_weights, axis=0))
    prox_weights = []
    for avg_w, global_w in zip(avg_weights, global_weights):
        prox_weights.append(avg_w + mu * (avg_w - global_w))
    return prox_weights