# Tests
- hello.ll.main.0.pbqpgraph - simple example of PBQP register allocation gaph.
- example.ll.high_register_pressure.0.pbqpgraph and ...1... - complex register allocation problem (not working)

# Generating examples
```bash
./generate.sh
```

In order to get solutuons LLVM should be patched.

# Results
high_register_pressure - is not working.

Rough time measurments for invalid GPU solver:
- high_register_pressure.1

GPU: ~160ms with loader and ~43ms without loader
LLVM: Wall time: 230ms, Sys time: 6ms (with `-time-passes` option)

## Failure of my heuristic
There are registers that cannot be spilled according to PBQP task.
This means that task potentially unsolvable after some RN reductions.

This means that heuristic for node selection is not a number of neighboring nodes, but the total cost of vector itself.

Additionally there is a problem in RNreduction (probably).

There is a problem with number of nodes in RN!!!!

There should be other reduction than simple full search.

