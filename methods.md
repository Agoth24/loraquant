# Implementation Plan

## methods

- svd operation
- sublora split

- compute h from rho

- create high and low bit matrices for B and A
- STE optimization for high bits
- STE optimization for low bits

- quantizers
    - rtn for high-bit
    - binary for low-bit


### STE optimizer

- loop over steps
    - quantize B and A
    - dequantize them
    - compute loss & backpropagate w/ STE
        - B and A are updated


