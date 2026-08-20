# Prebuilt Block-Sparse-Attention wheel

File:

```text
block_sparse_attn-0.0.2+cu124torch2.6sm80ptx1-cp311-cp311-linux_x86_64.whl
```

SHA-256:

```text
1a8f3e7bf45f25dba2c2df83c9ecf13601b27ea5cdcea6e4283a1539e72f8c7d
```

Compatibility:

- Linux x86_64
- CPython 3.11
- PyTorch 2.6.x with CUDA 12.4 wheels
- glibc 2.32 or newer
- libstdc++ with `GLIBCXX_3.4.29`
- NVIDIA GPU/driver capable of running the sm_80 cubin or compute_80 PTX

The target machine does not need a CUDA toolkit. The wheel was tested on an
RTX 4090 with PyTorch 2.6.0+cu124.

Build source: MIT HAN Lab Block-Sparse-Attention commit
`49d6c39e4dc0303442cda3bb758b3925d4399c49`, plus
`patches/block-sparse-attention-import-order.patch`.

Block-Sparse-Attention is distributed under the BSD 3-Clause License. The full
required notice is in `third_party/Block-Sparse-Attention-LICENSE`.
