# PBQP algorithm

## Run tests
Build
```bash
mkdir build
cd build
cmake ..
make
```

Run performance:
```bash
./Perf-measure (--use-GPU/--use-CPU/--use-heuristic) --in-file file-with-graph --out-file file-to-dump --check-solution
```

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

## Print graph
```bash
./Graph-print --in-file file-with-graph --out-file file-to-dump --LLVM
```

# Examples
```bash
./Graph-print --in-file ../tests/llvm/hello.ll.main.0.pbqpgraph --out-file out.dot --LLVM
./Perf-measure --use-heuristic --in-file ../tests/llvm/hello.ll.main.0.pbqpgraph --out-file hello-llvm-solution.out --LLVM --check-solution
/usr/local/cuda/bin/compute-sanitizer --tool memcheck  ./programm
```

Sanitizer didn't find any errors in false solution.
In simple test (hello world llvm) the answer is 0.
Sometimes reductions give non-zero answer:
```
reference:
graph Dump {
node[color=red, fontsize=14, style=filled, shape=oval]
"Solution:0" [color=coral, fontsize=18, style=filled, shape=oval]
"0x5dbe357f6b40" [label = "%14 1"]
"0x5dbe357a87e0" [label = "%10 2"]
"0x5dbe35919fc0" [label = "%8 2"]
"0x5dbe357a8880" [label = "%6 1"]
"0x5dbe35907f90" [label = "%5 1"]
"0x5dbe356ac260" [label = "%4 1"]
"0x5dbe357e5ea0" [label = "%2 1"]
"0x5dbe357a8880" -- "0x5dbe357a87e0"
"0x5dbe356ac260" -- "0x5dbe35919fc0"
}
reduction:
graph Dump {
node[color=red, fontsize=14, style=filled, shape=oval]
"Solution:10.0326" [color=coral, fontsize=18, style=filled, shape=oval]
"0x5dbe3590b280" [label = "%14 1"]
"0x5dbe3581a470" [label = "%10 0"]
"0x5dbe3590a3a0" [label = "%8 1"]
"0x5dbe357b56d0" [label = "%6 1"]
"0x5dbe357760b0" [label = "%5 1"]
"0x5dbe357a70d0" [label = "%4 2"]
"0x5dbe357a93b0" [label = "%2 1"]
"0x5dbe357b56d0" -- "0x5dbe3581a470"
"0x5dbe357a70d0" -- "0x5dbe3590a3a0"
}
Error: Differen answers: Ref[0.000000] GPU[10.032580]
```