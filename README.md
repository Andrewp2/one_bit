# ONE BIT

Here's the main idea: We take the backprop-free Evolution Guided General Optimization via Low-rank Learning (EGGROLL) algorithm, the architecture
from Tiny Recursive Model, we add a pruning/sparsity objective between blocks of neurons, and we keep the precision of the model in 1 bit (values in {-1, 1}), to maximize inference speed and minimize bandwidth costs.


Some benchmarks to try:

MNIST
CIFAR
Sudoku
ARC-AGI