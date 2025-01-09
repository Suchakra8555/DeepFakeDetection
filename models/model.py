import pickle

MODEL_PATHS = {
    "linear": "pickles/finalized_model_linear.sav",
    "svm": "pickles/finalized_model_svm.sav",
    "knn": "pickles/finalized_model_knn.sav"
}

def load_model(model_type):
    """Load the chosen model from pickle files."""
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Invalid model type. Choose from {list(MODEL_PATHS.keys())}")
    model_path = MODEL_PATHS[model_type]
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
