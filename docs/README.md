# emberserve documentation

Back to the [README](../README.md).

* [How it works](design.md): the step loop, scheduler, paged KV cache, prefix caching,
  attention backends, latent attention and MoE, CUDA graphs, int8, tensor parallelism,
  correctness.
* [Results in depth](results.md): ablations, the decode kernel, the benchmark correction,
  every sweep against vLLM, the gap analysis and fix history, Moonlight, real text, two API
  processes, the 7B tail.
* [Cold start](cold-start.md): A100 pod series and the Runpod Serverless series, phase by
  phase.
* [Models](models.md): the per-model table and a footnote on hosted APIs.
* [Running it](running.md): local, GPU, Runpod Serverless, benchmark commands.
* [GPU notes](gpu.md): pod setup and the pitfalls that cost GPU time.
* [Roadmap](roadmap.md): what is open.

Also: [scripts/](../scripts/README.md), [results/](../results/README.md),
[deploy/runpod/](../deploy/runpod/README.md).
