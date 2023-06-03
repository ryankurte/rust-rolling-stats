# rust-rolling-stats

Rolling statistics calculations (min/max/mean/std_dev) over arbitrary floating point numbers based
on Welford's Online Algorithm.

See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for more information

## Status

[![GitHub tag](https://img.shields.io/github/tag/ryankurte/rust-rolling-stats.svg)](https://github.com/ryankurte/rust-rolling-stats)
[![Build Status](https://travis-ci.com/ryankurte/rust-rolling-stats.svg?branch=master)](https://travis-ci.com/ryankurte/rust-rolling-stats)
[![Crates.io](https://img.shields.io/crates/v/drolling-stats.svg)](https://crates.io/crates/drolling-stats)
[![Docs.rs](https://docs.rs/drolling-stats/badge.svg)](https://docs.rs/drolling-stats)

## no_std

This crate is `no_std` compatible, simply disable the default features in your `Cargo.toml`:

```toml

[dependencies]
rolling-stats = { version = "0.6.0", default-features = false }

```
