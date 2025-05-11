def federated_adapt(weights_list, client_metrics, client_data_sizes, global_weights, prev_weights=None, round_num=0):
    total_data = sum(client_data_sizes)
    basic_weights = [size / total_data for size in client_data_sizes]
    num_clients = len(weights_list)

    if prev_weights is None or round_num == 0:
        adapted_weights = federated_avg(weights_list)
    else:
        adapted_weights = []
        for layer_idx in range(len(weights_list[0])):
            layer_weights = [weights[layer_idx] for weights in weights_list]
            layer_weights = np.array(layer_weights)

            client_performance = [client_metrics[i]['accuracy'] for i in range(num_clients)]
            performance_weights = np.array(client_performance) / np.sum(client_performance)

            client_data_weights = np.array(basic_weights)
            combined_weights = performance_weights * client_data_weights

            adapted_layer_weights = np.average(layer_weights, axis=0, weights=combined_weights)
            adapted_weights.append(adapted_layer_weights)

    return adapted_weights