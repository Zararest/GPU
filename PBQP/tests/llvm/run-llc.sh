# Debug version
CLANG_PATH=~/projects/llvm-project/llvm/build/bin

$CLANG_PATH/llc -pbqp-dump-graphs -regalloc=pbqp -O3 example.c