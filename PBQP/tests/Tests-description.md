# small-test
```
--nodes-num 7 --node-size 2 --avg-neighb-num 1 --num-of-cliques 5
```

# big-test
```
--nodes-num 70 --node-size 13 --avg-neighb-num 1 --num-of-cliques 30
```

# R0-R1-performance
--nodes-num 70 --node-size 32 --avg-neighb-num 1 --num-of-cliques 10

# from-paper
Node count: 41
Sample size: 13
Edge count: 31

This is a perfect match:
```
--nodes-num 41 --node-size 13 --avg-neighb-num 1 --num-of-cliques 10
```

Performance:
51 ms on load; 4 on calc

# from-paper bigger
Node count: 190
Sample size: 15
Edge count: 443

```
--nodes-num 190 --node-size 15 --avg-neighb-num 1 --num-of-cliques 1
```
edges: 189

Performance:
48 ms on load; 26 on calc

# from-paper-bigger-with-loops
Node count: 190
Sample size: 15
Edge count: 443

```
--nodes-num 190 --node-size 15 --avg-neighb-num 3.5 --num-of-cliques 20
```
edges: 432

Performance:
```
Profile info:
        Loader: 48.504000
        Loop header iter num: 2
        R0: 6.466000
        R1: 0.315000
        R1: 0.166000
        R0: 0.278000
        Clean up: 0.339000
        Pass 6: 0.000000
        Loop end

        Loop header iter num: 71
        RN: 14.171000
        R1: 7.493000
        R0: 5.311000
        R1: 5.777000
        R1: 4.724000
        R0: 5.353000
        Clean up: 5.473000
        Pass 14: 0.000000
        Loop end

        Final full search with RN: 0.000000
        Deleter: 0.149000

Reductions on GPU time: 104.997000ms
```