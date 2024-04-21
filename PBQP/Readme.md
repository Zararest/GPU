# PBQP algorithm

## Reduction pass
All reductions accepts `GPUResult` class with GPU graph and `Solution` class. 
They also returns the same class, but with extended solution.
If reduction changes graph, it edits only adjustment matrix without changing its size.
In order to remove empty nodes `Cleanup` pass should be run.

### Cleanup pass
This pass finds nodes which have -1 in a AdjMatrix[i][i] cell and removes them.

### R0 reduction
Finds nodes without neighbors and removes them adding best cost to the silution.
![img](../img/R0.jpg)

## GPU evaluation
```bash
nvprof ./program
```