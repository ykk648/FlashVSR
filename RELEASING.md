# Fork release checklist

Model weights, runtime outputs, virtual environments, caches, and built wheels
are intentionally ignored by Git. Do not force-add them to a commit.

## Source release

Run the local checks:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q flashvsr_api scripts
git diff --check
git status --short
```

Confirm the remote before pushing. A fresh fork normally uses `upstream` for
the original project and `origin` for your fork:

```bash
git remote -v
git remote rename origin upstream
git remote add origin https://github.com/OWNER/FlashVSR.git
git push -u origin main
```

Replace `OWNER` with the fork owner. Never run the remote commands without
checking their output first.

## Block-Sparse-Attention wheel

The prebuilt wheel is 62 MiB and platform-specific, so publish it as a GitHub
Release asset instead of committing it to Git history.

Release asset:

```text
block_sparse_attn-0.0.2+cu124torch2.6sm80ptx1-cp311-cp311-linux_x86_64.whl
```

Expected SHA-256:

```text
1a8f3e7bf45f25dba2c2df83c9ecf13601b27ea5cdcea6e4283a1539e72f8c7d
```

After configuring the fork remote, a maintainer can create the release with:

```bash
gh release create block-sparse-attn-cu124-torch2.6 \
  dist/block_sparse_attn-0.0.2+cu124torch2.6sm80ptx1-cp311-cp311-linux_x86_64.whl \
  --title "Block-Sparse-Attention for Torch 2.6 cu124" \
  --notes-file third_party/Block-Sparse-Attention-WHEEL.md
```

The wheel was built from MIT HAN Lab Block-Sparse-Attention commit
`49d6c39e4dc0303442cda3bb758b3925d4399c49` with
`patches/block-sparse-attention-import-order.patch`. Binary redistribution is
covered by `third_party/Block-Sparse-Attention-LICENSE`, which must remain in
source releases and release notes.
