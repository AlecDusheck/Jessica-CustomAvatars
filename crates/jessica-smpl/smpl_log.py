import pickle
import numpy as np
from scipy import sparse

def log_tensor_size(name, tensor):
    if sparse.issparse(tensor):
        print(f"{name}: Sparse tensor of shape {tensor.shape}, with {tensor.nnz} non-zero elements")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}: Dense tensor of shape {tensor.shape}, dtype {tensor.dtype}")
    elif isinstance(tensor, str):
        print(f"{name}: String value: '{tensor}'")
    elif hasattr(tensor, 'r'):  # Check if it's a Chumpy object
        r = tensor.r
        if isinstance(r, np.ndarray):
            print(f"{name}: Chumpy object with underlying ndarray of shape {r.shape}, dtype {r.dtype}")
        else:
            print(f"{name}: Chumpy object with non-ndarray .r attribute")
    else:
        print(f"{name}: {type(tensor).__name__} value")

def get_size(tensor):
    if hasattr(tensor, 'nbytes'):
        return tensor.nbytes
    elif hasattr(tensor, 'r') and hasattr(tensor.r, 'nbytes'):
        return tensor.r.nbytes
    return 0

def main():
    with open('SMPL_NEUTRAL.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print("SMPL Data Contents:")
    for key, value in data.items():
        log_tensor_size(key, value)

    total_size = sum(get_size(value) for value in data.values())
    print(f"\nTotal size of numeric tensors: {total_size} bytes")

if __name__ == "__main__":
    main()