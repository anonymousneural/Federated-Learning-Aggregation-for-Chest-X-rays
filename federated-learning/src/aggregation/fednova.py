def federated_nova(weights_list, global_weights, client_updates_norm=None, tau=None):
    if tau is None:
        tau = CONFIG['tau']
    
    nova_weights = []
    
    avg_weights = []
    for layer_weights in zip(*weights_list):
        avg_weights.append(np.mean(layer_weights, axis=0))
    
    if client_updates_norm is None:
        client_updates_norm = []
        for client_weights in weights_list:
            norm = np.linalg.norm(client_weights - global_weights)
            client_updates_norm.append(norm)
    
    for avg_w, global_w, norm in zip(avg_weights, global_weights, client_updates_norm):
        if norm > 0:
            normalized_update = (avg_w - global_w) / norm
            nova_weights.append(global_w + tau * normalized_update)
        else:
            nova_weights.append(avg_w)
    
    return nova_weights