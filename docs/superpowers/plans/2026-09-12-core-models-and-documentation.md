# Core models and documentation

Tracking issue: [#8](https://github.com/newpoluton-alt/RustTorch/issues/8).

The requested direction is complete framework functionality in RustTorch, with
core models and training first. This contribution advances a coherent model
and training slice and improves the existing user's path from installation to
training. The larger [roadmap](../../roadmap.md) remains open.

## Implementation

1. Replace crate landing-page compatibility inventories with Rust examples for
   tensors, training, inference, persistence, and datasets.
2. Rewrite the GitHub README and add a training guide; retain exact provenance
   and coverage in dedicated compatibility pages.
3. Add validated fallible convolution, layer normalization, and embedding
   configurations and modules, reusing LibTorch numerical operations. Expose
   the existing parameter store through the RustTorch namespace.
4. Add AdamW, RMSprop, learning-rate adjustment, and gradient clipping using the
   existing runtime and validation rules.
5. Complete DataLoader documentation, examples, benchmark methodology and CI
   gaps, including a precise checkpoint capability matrix.
6. Update the ledger, generated pages, changelog and upstream references; run
   focused Rust checks, multi-step CPU parity, all workspace checks, rustdoc,
   examples, policy tests, package inspection, and backend checks.

## Acceptance and limits

New public items have useful rustdoc, valid examples, rejected invalid inputs,
and focused executable evidence. Numerical claims identify the pinned reference
and backend. No new dependency or unsafe boundary is introduced. Existing
public call patterns continue to work. The complete API census, remaining core
layer and optimizer families, compilation, distributed training, and modality
packages remain separate milestones rather than silently acquiring a support
claim from this change.

Contribution is on `codex/rusttorch-docs-and-models`; commits require DCO sign-off
and review follows the repository PR template. Local checks cannot establish
unavailable accelerator/platform or published-package evidence.
