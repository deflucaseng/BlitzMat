# BlitzMat

A C/C++ based OpenCL library which utilizes the power of parallel computing to perform NumPy based matrix operations.



## Features

* Easy Python UI for NumPy Matrix Multiplications
* Accomodation for CPU/GPU setups
* Lightweight Codebase

## Installation



```bash
# Simple installation
pip install blitzmat

# Installation from source
git clone https://github.com/deflucaseng/blitzmat.git
cd project
pip install -e .
```

## Quick Start

Simple Python Library based UI:

```python
import numpy as np
from blitzmul import blitz


# For basic matrix multiplication

A = np.array([[1, 2],
			  [3, 4]])

B = np.array([[5, 6],
			  [7, 8]])

C = blitz.mat_mul(A, B)

print(C)
```



## Configuration

Explain any configuration options and how to use them:

```python
# Example configuration file
from blitzmat import blitz

blitz.set_device("CPU") #Searches for an available CPU and utilizes it for computation

blitz.set_device("GPU") #Searches for an available GPU and utilizes it for parallel computation

```
## Usage Guide

Starter Code:
```python
from blitzmat import blitz
import numpy

blitz.set_device("GPU")  #Note it defaults to CPU

A = np.array([[1, 2],
			  [3, 4]])

B = np.array([[5, 6],
			  [7, 8]])

```




Supported operations are as follows



### Matrix Multiplication
```python
C = blitz.mat_mul(A, B)
```

### Element-Wise Operations
```python
C = blitz.elemwise_add(A, B) # Addition

D = blitz.elemwise_sub(A, B) # Subtraction

E = blitz.elemwise_mul(A, B) # Multiplication

F = blitz.elemwise_div(A, B) # Division
```

### Transpose
```python
C = blitz.transpose(A)
```

### Inverse
```python
C = blitz.inverse(A)
```

### Determinant
```python
C = blitz.determinant(A)
```

### Trace
```python
C = blitz.trace(A)
```

### Frobenius Norm
```python
C = blitz.frb_nrm(A)
```

<!--### Eigenvalues/Eigenvectors
```python
lstscalars, lstvectors = blitz.eig(A)
```
-->
















## Acknowledgments

* This project utilized the OpenCL library

## Contact

* Maintainer: [Lucas Eng](https://github.com/deflucaseng)
* Email: intlucaseng@gmail.com
* My [LinkedIn](https://www.linkedin.com/in/lucas-eng/)

## Citation

If this project helps your research, please cite it as:

```bibtex
@software{Numerikal Labs,
  author = {Lucas Eng},
  title = {BlitzMat},
  year = {2024},
  url = {https://github.com/deflucaseng/BlitzMat.git}
}
```
