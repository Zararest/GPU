# Tests
- hello.ll.main.0.pbqpgraph - simple example of PBQP register allocation gaph.
- example.ll.high_register_pressure.0.pbqpgraph and ...1... - complex register allocation problem (not working)

# Generating examples
```bash
./generate.sh
```

Every BB is converted into one file.

# Results
high_register_pressure - is not working.

Rough time measurments for invalid GPU solver:
- high_register_pressure.1

GPU: ~160ms with loader and ~43ms without loader
LLVM: Wall time: 230ms, Sys time: 6ms (with `-time-passes` option)

