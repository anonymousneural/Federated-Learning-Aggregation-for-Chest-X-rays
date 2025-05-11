def save_model_weights(model, filepath):
    model.save_weights(filepath)

def load_model_weights(model, filepath):
    model.load_weights(filepath)

def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f)

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)