SWIM: Sample Where It Matters!
==============================

This repository implements swimnetworks (see original [repo](https://gitlab.com/felix.dietrich/swimnetworks) and [paper](https://arxiv.org/abs/2306.16830)) in Rust.

## Why in Rust?

Inspired by the [Burn framework](https://github.com/tracel-ai/burn?tab=readme-ov-file#getting-started), it is possible to use components that support [no_std](https://docs.rust-embedded.org/book/intro/no-std.html). Current implementation does not offer other backend options than statically build openblas, but in future there can be more backend options to increase portability and potentially run on embedded devices.

## Examples

For simple regression and classification examples see the [examples folder](https://github.com/AlphaGergedan/swimnetworks-rs/tree/master/examples).

## Status

Current implementation almost covers all the features from the original library, but there is still no guarantee at this stage if you decide to use this crate.
