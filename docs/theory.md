# Theory notes

Working notes on the two reference papers and on the numerical choices made in
this repository. Formulas use the conventions of the code.

## 1. Rough Bergomi (Bayer, Friz, Gatheral 2016)

### Why rough volatility

Two empirical facts motivate the model:

* **Realised volatility is rough.** Log-volatility behaves like a fractional
  Brownian motion with Hurst exponent `H ≈ 0.1` (Gatheral, Jaisson, Rosenbaum
  2018), far rougher than the `H = 1/2` of a diffusion.
* **The at-the-money skew explodes at short maturities.** On SPX the skew
  `ψ(T) = |∂_k σ_BS(k = 0, T)|` follows a power law `T^(-0.4)`, roughly.
  Classical stochastic volatility models give a skew that is flat as `T → 0`;
  they can only fit the short end by adding jumps or many factors.

A model driven by a fractional kernel with exponent `H − 1/2` produces a skew
of order `T^(H − 1/2)`: one parameter explains both facts.

### The model

Under the pricing measure, with zero rates:

```
dS_t / S_t = sqrt(v_t) dZ_t
v_t        = ξ0(t) · exp( η · sqrt(2H) · Y_t − η²/2 · t^(2H) )
Y_t        = ∫_0^t (t − s)^(H − 1/2) dW_s
dZ_t       = ρ dW_t + sqrt(1 − ρ²) dW⊥_t
```

| Parameter | Role | Effect on the surface |
|---|---|---|
| `H ∈ (0, 1/2]` | roughness of the variance paths | how fast the skew decays with maturity |
| `η > 0` | volatility of variance | smile curvature and overall skew size |
| `ρ ∈ [−1, 1]` | spot–variance correlation | sign and size of the skew |
| `ξ0(t)` | forward variance curve, `ξ0(t) = E[v_t]` | level (here flat: a single number) |

`sqrt(2H) · Y_t` is a Riemann–Liouville fractional Brownian motion with
variance `t^(2H)`, so the exponential is a martingale in `η` and `E[v_t] = ξ0(t)`:
the model is consistent with the variance swap curve by construction. Bayer,
Friz and Gatheral fit SPX smiles with just three numbers, typically
`H ≈ 0.07`, `η ≈ 1.9`, `ρ ≈ −0.9`.

### Why it is expensive

`Y` is not a semimartingale and not Markovian: `v_t` depends on the whole past
of `W`. There is no PDE, no characteristic function, no closed formula. Prices
come from Monte Carlo, and a calibration needs hundreds of them. This is the
bottleneck the neural network removes.

### Simulation: the hybrid scheme (Bennedsen, Lunde, Pakkanen 2017)

A plain Riemann sum for `Y` fails because the kernel `(t − s)^(H − 1/2)` blows
up at `s = t`. The hybrid scheme splits the integral on a grid of step `Δ`:

```
Y_{t_i} ≈ ∫_{t_{i−1}}^{t_i} (t_i − s)^α dW_s            exact, where the kernel is singular
        + Σ_{k=2}^{i} (b_k Δ)^α · (W_{t_{i−k+1}} − W_{t_{i−k}})   step function elsewhere
```

with `α = H − 1/2` and the optimal evaluation points
`b_k^α = (k^(α+1) − (k−1)^(α+1)) / (α+1)`, the average of the kernel over each cell.

The first term and the Brownian increment over the same step are jointly
Gaussian with covariance

```
Var(ΔW) = Δ      Cov = Δ^(α+1) / (α+1)      Var(local) = Δ^(2α+1) / (2α+1)
```

so they are sampled exactly from two independent normals. The second term is a
discrete convolution of past increments with fixed weights, implemented as one
matrix product. For `H = 1/2` every weight is one and `Y = W` exactly, which
gives a path-by-path unit test.

The spot is then advanced by log-Euler. The whole simulator is written with
differentiable PyTorch operations, so pathwise sensitivities with respect to
`(H, η, ρ, ξ0)` are available by automatic differentiation (phase 2).

### Variance reduction: the mixed estimator

Conditionally on `W`, the variance path is known and `S_T` is lognormal. The
noise `W⊥` can therefore be integrated out analytically (Romano–Touzi; used for
rough Bergomi by McCrickerd and Pakkanen 2018):

```
price = E[ Black( S0 · exp(ρ I_T − ρ²/2 · V_T),  K,  sqrt((1 − ρ²) V_T) ) ]
I_T = ∫_0^T sqrt(v) dW          V_T = ∫_0^T v dt
```

Three consequences matter here: the variance is lower, no spot path has to be
simulated, and the estimate is a smooth, strictly positive function of the
strike, so the implied volatility always exists. On top of this the code uses
antithetic paths and the conditional forward `S0 · exp(ρ I_T − ρ²/2 · V_T)`,
whose mean is exactly `S0`, as a control variate.

### Checks available without a closed formula

* `E[S_t] = S0` and `E[v_t] = ξ0` for all `t`.
* `η = 0` is Black–Scholes; `H = 1/2` makes `Y` a Brownian motion.
* Published Monte Carlo prices (Bayer, Ben Hammouda, Tempone 2020, Table 5.1).
* The short-maturity skew (Alòs, León, Vives 2007; Fukasawa 2017):

  ```
  ∂_k σ_BS(0, T) ≈ ρ η sqrt(2H) / (2 (H + 1/2)(H + 3/2)) · T^(H − 1/2)
  ```

  which reduces to `ρ η / 4` for `H = 1/2`, the known skew of a lognormal
  stochastic volatility model whose variance has volatility `η`.

## 2. Deep calibration (Horvath, Muguruza, Tomas 2021)

### Two steps instead of one

Calibration solves `θ* = argmin_θ ‖σ_model(θ) − σ_market‖`. The slow part is
the map `θ → σ_model(θ)`. The paper's proposal:

1. **Offline:** learn the pricing map `θ → σ_model(θ)` with a neural network,
   from surfaces generated by the slow pricer.
2. **Online:** calibrate with a standard optimiser, using the network instead
   of the pricer.

The alternative, learning the inverse map `σ_market → θ` directly (Hernandez
2017), is rejected for reasons that matter in practice:

* The network is trained on **synthetic** data only. It never sees market data,
  so there is no risk of fitting a market regime, and the training set can be
  as large as the compute budget allows.
* The model is unchanged: the network is a **numerical approximation of a
  known function**, so its error can be measured against the original pricer
  at any point. This is what makes validation, and this project, possible.
* The calibration objective (weights, subsets of quotes, bid-ask) can change
  without retraining.

### Image-based learning

The network outputs the whole surface on a fixed grid, like the pixels of an
image, rather than one volatility for inputs `(θ, T, K)`. The output dimension
is larger, but the network exploits the structure across neighbouring points
and the training set needs one Monte Carlo run per sample, shared by all
strikes and maturities.

### Their set-up, and what is different here

| | Horvath et al. | This repository |
|---|---|---|
| Parameter sampling | uniform | scrambled Sobol |
| Grid | 8 maturities × 11 strikes, strikes 0.5–1.5 at every maturity | 8 maturities × 10 strikes, strikes scaled by `sqrt(T)` |
| Network | 4 hidden layers × 30, ELU | 4 hidden layers × 64, SiLU |
| Input / output scaling | parameters to `[−1, 1]`, vols standardised | same |
| Optimiser for calibration | Levenberg–Marquardt | trust-region reflective (bounded LM) |
| Evaluation | in-sample domain | in-domain, interpolation gap, out-of-domain |

Two deliberate departures:

* **Strikes scaled by `sqrt(T)`.** A strike of 0.5 at a one-month maturity is
  tens of standard deviations away; its Monte Carlo implied volatility is
  noise, and fitting noise costs network capacity.
* **Evaluation outside the training domain.** The paper shows that the network
  is accurate where it was trained. The question asked here is the
  complementary one.

### What the approximation can and cannot do

The network is an interpolator of a smooth function on a compact box. Inside
the box, with enough samples, the error is limited by the Monte Carlo noise of
the training labels, and averaging over many noisy samples lets the network do
*better* than any single label. Outside the box nothing constrains it: the
function it has learned is whatever the last layers extrapolate to. Calibration
inherits this: the optimiser will happily return parameters for a surface the
network cannot represent, and only the fit error and the bounds reveal it.

## References

* C. Bayer, P. Friz, J. Gatheral (2016). *Pricing under rough volatility.* Quantitative Finance 16(6).
* B. Horvath, A. Muguruza, M. Tomas (2021). *Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models.* Quantitative Finance 21(1). arXiv:1901.09647.
* M. Bennedsen, A. Lunde, M. Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics 21(4).
* R. McCrickerd, M. Pakkanen (2018). *Turbocharging Monte Carlo pricing for the rough Bergomi model.* Quantitative Finance 18(11).
* C. Bayer, C. Ben Hammouda, R. Tempone (2020). *Hierarchical adaptive sparse grids and quasi-Monte Carlo for option pricing under the rough Bergomi model.* Quantitative Finance 20(9). arXiv:1812.08533.
* J. Gatheral, T. Jaisson, M. Rosenbaum (2018). *Volatility is rough.* Quantitative Finance 18(6).
* E. Alòs, J. León, J. Vives (2007). *On the short-time behavior of the implied volatility for jump-diffusion models with stochastic volatility.* Finance and Stochastics 11(4).
* M. Fukasawa (2017). *Short-time at-the-money skew and rough fractional volatility.* Quantitative Finance 17(2).
