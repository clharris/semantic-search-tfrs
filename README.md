# semantic-search-tfrs
Using TF Recommenders to build a two-towers model for semantic product search and ranking.

This
[Reference paper](https://dl.acm.org/doi/pdf/10.1145/3292500.3330759)

The data used here is the [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for
Improving Product Search](https://arxiv.org/pdf/2206.06588.pdf) which can be downloaded as part of this
[AICrowd challenge](https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search). The data has 
been filtered to only the `us` locale to trim down size and reduce complexity.

This has been tested on Google Compute Platform on a `tf-2-11-gpu-debian-10` family type of [Deep learning VM](https://cloud.google.com/deep-learning-vm/docs/)


