from .opencl_ops import OperationManager

# Operation types for multi-vector operations
OPERATIONS = {
    'add': 'add',
    'subtract': 'subtract',
    'multiply': 'multiply',
    'divide': 'divide',
    'matrix_multiply': 'matrix_multiply'
}

# Operation types for single-vector operations
SINGLE_OPERATIONS = {
    'transpose': 'transpose',
    'inverse': 'inverse',
    'trace': 'trace',
    'frobenius_norm': 'frobenius_norm',
    'determinant': 'determinant'
}

# Device types
DEVICES = {
    'CPU': 'CPU',
    'GPU': 'GPU'
}

__all__ = [
    'OperationManager',
    'OPERATIONS',
    'SINGLE_OPERATIONS',
    'DEVICES'
]