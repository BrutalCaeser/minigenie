# Foundations: From First Principles to Video World Models

This document builds your understanding from the ground up. Each section assumes you've read the previous one. There are no skips. If you understand everything here, you can explain your project on a whiteboard in an interview without looking anything up.

---

## Part 1: Why we need latent representations at all

A 128×128 RGB image is a point in a 49,152-dimensional space. Most points in that space look like TV static. Real images occupy an incredibly thin manifold — a low-dimensional surface winding through that high-dimensional space.

Consider all possible 128×128 frames from CoinRun. They vary along a small number of meaningful axes: character position (x, y), enemy positions, platform layout, background scroll offset, animation frame. Maybe 50-100 real degrees of freedom. But the pixel representation uses 49,152 numbers.

Predicting the next frame directly in pixel space means your model must simultaneously get right every single one of those 49,152 values. If it's uncertain about whether a distant coin is 3 pixels left or 4 pixels left, MSE loss punishes it, so it hedges by producing a blurry average. This is why direct pixel prediction produces blurry results — it's not a bug, it's the mathematically optimal response to uncertainty under squared error loss.

The solution: learn a compressed representation where the meaningful axes of variation are separated from irrelevant details, then do prediction in that compressed space.

---

## Part 2: Autoencoders — learning to compress

An autoencoder is two networks: an encoder $E$ that maps input $x$ to a compact code $z = E(x)$, and a decoder $D$ that reconstructs from the code $\hat{x} = D(z)$. Training minimizes reconstruction error $\|x - D(E(x))\|^2$.

The bottleneck (making $z$ much smaller than $x$) forces the encoder to learn which information matters. If $x$ is 49,152 dimensions and $z$ is 256 dimensions, the encoder must discard 99.5% of the input — and it learns to keep the meaningful parts.

But plain autoencoders have a problem: the latent space has no structure. Two similar images might map to distant points in $z$-space, and interpolating between two codes might produce garbage. The latent space is a tangled mess, useful for compression but not for generation.

---

## Part 3: VAEs — making the latent space structured

Variational Autoencoders (Kingma & Welling, 2013) fix this by imposing structure on the latent space. Instead of mapping $x$ to a single point $z$, the encoder outputs parameters of a distribution: $\mu(x)$ and $\sigma(x)$. During training, $z$ is sampled from $\mathcal{N}(\mu(x), \sigma^2(x))$ using the reparameterization trick:

$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The trick: writing $z$ this way makes $z$ a deterministic function of $\mu$, $\sigma$, and $\epsilon$, so gradients flow through to the encoder even though sampling is involved.

The VAE loss has two terms:

$$\mathcal{L}_{VAE} = \underbrace{\|x - D(z)\|^2}_{\text{reconstruction}} + \underbrace{\beta \cdot D_{KL}[\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)]}_{\text{regularization}}$$

The KL term pushes the encoder's output distribution toward the standard normal. This creates structure: codes for similar images cluster together, the space is smooth (interpolation works), and you can generate new samples by decoding random points from $\mathcal{N}(0, I)$.

**The tension:** Reconstruction wants $z$ to be as informative as possible (high variance, far from standard normal). Regularization wants $z$ to be as close to standard normal as possible (destroying information). The balance $\beta$ controls this tradeoff. Too much regularization → blurry reconstructions. Too little → unstructured latent space.

---

## Part 4: VQ-VAE — discrete codes instead of continuous distributions

Vector Quantized VAE (van den Oord et al., 2017) takes a different approach to structuring the latent space. Instead of forcing a Gaussian prior, it uses a discrete codebook.

**The codebook** is a learnable table of $K$ vectors, each $D$-dimensional:
$$\mathcal{C} = \{e_1, e_2, ..., e_K\}, \quad e_k \in \mathbb{R}^D$$

Think of it as a palette of 512 "concept tiles." Every frame gets described as a mosaic of tiles from this palette.

**Encoding:** The encoder outputs a continuous feature map $z_e(x)$ at each spatial position. For a 128×128 input with 4× downsampling, this is a 32×32 grid (or 16×16 with 8× downsampling), where each grid cell is a $D$-dimensional vector.

**Quantization:** Each continuous vector gets replaced by the nearest codebook entry:
$$z_q = e_{k^*}, \quad \text{where } k^* = \arg\min_k \|z_e - e_k\|^2$$

This is just nearest-neighbor lookup. The output is a grid of integers (codebook indices).

**Decoding:** The decoder takes the grid of codebook vectors and reconstructs the image.

**The gradient problem:** Quantization (argmin) has zero gradient everywhere. You can't backpropagate through a lookup table. The solution is the straight-through estimator: during the forward pass, use $z_q$ (the quantized vector). During the backward pass, pretend quantization didn't happen and copy the gradient from $z_q$ directly to $z_e$:

$$\text{Forward: } z_q = \text{sg}(z_q - z_e) + z_e$$
$$\text{Backward: } \nabla_{z_e} \mathcal{L} = \nabla_{z_q} \mathcal{L}$$

where $\text{sg}(\cdot)$ is the stop-gradient operator.

**The VQ-VAE loss:**
$$\mathcal{L} = \underbrace{\|x - D(z_q)\|^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}(z_e) - z_q\|^2}_{\text{codebook learning}} + \underbrace{\beta \|z_e - \text{sg}(z_q)\|^2}_{\text{commitment loss}}$$

Term 1 trains the encoder and decoder to reconstruct well. Term 2 moves codebook entries toward encoder outputs (trains the codebook). Term 3 prevents the encoder from moving its outputs too fast for the codebook to follow (the encoder "commits" to staying near the codebook).

In practice, Term 2 is replaced by **Exponential Moving Average (EMA) updates:** instead of gradient descent, codebook vectors are updated as running averages of the encoder outputs that map to them:

$$e_k \leftarrow \gamma \cdot e_k + (1 - \gamma) \cdot \bar{z}_{e,k}$$

where $\bar{z}_{e,k}$ is the mean of all encoder outputs assigned to code $k$ in the current batch, and $\gamma = 0.99$. This is more stable than gradient-based codebook learning.

**Why discrete over continuous?** Several reasons. Discrete tokens are natural inputs for transformer-based dynamics models (treat frames as token sequences, like language). Discrete representations avoid the "blurry average" problem — the codebook forces a hard choice between distinct options. And information-theoretically, $K=512$ codebook entries with a 16×16 grid gives $512^{256}$ possible representations — an astronomically large space that can represent enormous visual diversity despite being discrete.

**Codebook collapse:** The major failure mode. If the encoder output distribution shifts during training, some codebook entries become "dead" — no encoder output is nearest to them, so they never get updated, so they drift further from the encoder distribution, so they get used even less. A positive feedback loop that kills codebook utilization.

Fixes (use all three):
1. **EMA updates** (described above) — more stable than gradient descent
2. **Codebook reset** — periodically replace dead entries with random encoder outputs from the current batch
3. **L2 normalization** — project both encoder outputs and codebook entries onto the unit sphere before computing distances. This prevents the encoder from "escaping" the codebook by increasing its output magnitude.

---

## Part 5: How images are generated — the problem flow matching solves

Given a dataset of images, we want to learn a model that can generate new images from the same distribution. Mathematically: given samples from $p_{data}(x)$, learn to sample from $p_{data}$.

The core insight behind both diffusion models and flow matching: it's easy to sample from a simple distribution (Gaussian noise). If we can learn a transformation from noise to data, we can generate images by sampling noise and applying the transformation.

**The question is: what transformation?**

**Diffusion models** frame this as reversing a noising process. You progressively add noise to data until it becomes pure Gaussian noise (forward process), then learn to reverse each step (backward process). The learned model is a denoiser — given a noisy image at noise level $\sigma$, predict the clean image (or equivalently, predict the noise).

**Flow matching** frames this as learning a velocity field. Imagine every data point $x_1$ paired with a noise point $x_0 \sim \mathcal{N}(0, I)$. Define a straight path from $x_0$ to $x_1$ parameterized by time $t \in [0, 1]$:

$$x_t = (1 - t) \cdot x_0 + t \cdot x_1$$

At $t = 0$: pure noise. At $t = 1$: clean data. At intermediate $t$: a blend.

The velocity (how fast and in what direction you need to move along this path) is:

$$\frac{dx_t}{dt} = x_1 - x_0$$

This is constant along each path — the velocity is just "target minus noise."

**The flow matching training objective:** teach a neural network $v_\theta$ to predict this velocity:

$$\mathcal{L}_{FM} = \mathbb{E}_{t \sim U[0,1], \, x_1 \sim p_{data}, \, x_0 \sim \mathcal{N}(0,I)} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right]$$

In words: sample a data point, sample noise, pick a random time, interpolate to get $x_t$, predict the velocity, and penalize the prediction error.

**Why this works (the non-obvious part):** Each training example gives the model a *conditional* velocity — the velocity for one specific noise-data pair. But at test time, we need the *marginal* velocity — the average velocity at each point over all possible data that could have contributed. The remarkable theorem (Lipman et al., 2023) is that the marginal velocity field (what we want) can be learned by regressing conditional velocity fields (what we compute per sample). The conditional flow matching loss is a valid Monte Carlo estimate of the marginal flow matching loss. This is analogous to how denoising score matching learns the score function via per-sample denoising targets.

**Inference (generating a new image):** Start from noise $x_0 \sim \mathcal{N}(0, I)$. Integrate the learned velocity field from $t=0$ to $t=1$ using Euler steps:

$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

With $N$ steps and $\Delta t = 1/N$, you take $N$ small steps from noise to data. More steps = higher quality but slower. Typically $N = 10$-$20$ is sufficient for flow matching (vs 20-50 for DDPM-style diffusion) because the straight paths are easier to integrate than the curved paths of diffusion.

**Conditional flow matching (what we use for world modeling):** The velocity model takes additional conditioning: $v_\theta(x_t, t, c)$ where $c$ includes context frames and the action. The loss stays the same — condition the model, but the target velocity is still $x_1 - x_0$. The conditioning tells the model *which* data distribution to target (the distribution of next frames given this context and action), and the velocity field transports noise to samples from that conditional distribution.

---

## Part 6: Making it a world model — putting the pieces together

A video world model is a conditional generative model where:

- **What we condition on:** past frames and current action
- **What we generate:** the next frame

The training data is sequences $(o_1, a_1, o_2, a_2, ..., o_T)$ where $o_t$ is an observation (frame) and $a_t$ is the action taken at time $t$.

**The full forward pass during training:**

1. Take a window of frames: $o_{t-3}, o_{t-2}, o_{t-1}, o_t$ (context) and $o_{t+1}$ (target)
2. The action $a_t$ was taken between $o_t$ and $o_{t+1}$
3. Sample time $t_{flow} \sim U[0,1]$ and noise $\epsilon \sim \mathcal{N}(0, I)$
4. Interpolate: $x_{t_{flow}} = (1 - t_{flow}) \cdot \epsilon + t_{flow} \cdot o_{t+1}$
5. Concatenate context frames channel-wise with $x_{t_{flow}}$ as input to the U-Net
6. Inject action $a_t$ via adaptive normalization
7. U-Net predicts velocity $v_\theta$
8. Loss: $\|v_\theta - (o_{t+1} - \epsilon)\|^2$

**At inference (interactive generation):**

1. Start with real context frames: $o_{t-3}, o_{t-2}, o_{t-1}, o_t$
2. User selects action $a_t$
3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
4. Run $N = 15$ Euler steps: $x_0 = \epsilon$, repeatedly $x \leftarrow x + \frac{1}{N} \cdot v_\theta(x, t, \text{context}, a_t)$
5. Final $x$ is the predicted next frame $\hat{o}_{t+1}$
6. Shift context window: new context is $o_{t-2}, o_{t-1}, o_t, \hat{o}_{t+1}$
7. User selects next action, repeat

**The critical problem — distribution shift:** During training, context frames are always real (from the dataset). During inference, after the first step, context frames include model predictions. The model has never seen its own errors as input. Small prediction errors compound: each imperfect frame makes the next prediction slightly worse, and after 15-30 steps, the output diverges from anything realistic.

**GameNGen's fix — noise augmentation:** During training, randomly add Gaussian noise to the context frames. This teaches the model to handle imperfect inputs — because noisy context frames are a decent approximation of what imperfect predictions look like. With noise augmentation, the model learns to "clean up" its inputs while predicting the next frame, dramatically slowing the degradation during autoregressive rollout.

---

## Part 7: How action conditioning actually works in a U-Net

The U-Net is a convolutional encoder-decoder with skip connections (the U shape). The question: how do we make it care about the action?

**Adaptive Group Normalization (AdaGN)** — the standard approach (used in DIAMOND, Stable Diffusion, DiT):

Every ResBlock in the U-Net contains a normalization step. Normally, GroupNorm normalizes features to zero mean and unit variance. AdaGN adds learned scaling and shifting *that depend on the conditioning*:

```
# Standard GroupNorm:
x_norm = (x - mean) / std

# AdaGN:
scale, shift = Linear(cond_embedding).chunk(2)   # condition → scale & shift
x_norm = GroupNorm(x) * (1 + scale) + shift       # modulate normalized features
```

The conditioning embedding combines the flow time $t$ and the action $a$:

```
time_emb = MLP(sinusoidal_embed(t))      # [B, 256]
action_emb = MLP(one_hot(action))        # [B, 256]
cond_emb = time_emb + action_emb         # [B, 256]
```

This is injected into every ResBlock in the U-Net. The model can choose to amplify or suppress different feature channels depending on the action. For example, when the action is "move right," it might amplify features detecting rightward motion and suppress features for leftward motion.

**Why this works:** The action doesn't change what's in the image — it changes how the image should change. AdaGN modulates the network's internal processing based on the action, which is exactly the right inductive bias. It says "process these visual features, but adjust your processing based on what action was taken."

---

## Part 8: The evaluation questions that matter

Once the system works, the interesting questions are not "does it generate pretty pictures" but:

**Does the model actually use actions?** Generate 8 continuations from the same starting frame using all 8 possible actions. If the outputs differ meaningfully (character moves in different directions), the model is action-conditioned. If they look the same, action conditioning failed.

**How fast does quality degrade?** Plot PSNR (peak signal-to-noise ratio) between predicted frames and ground truth frames at each step of a rollout. A well-functioning model degrades gracefully; a poorly-functioning one falls off a cliff.

**What physical regularities does it learn?** Does the character obey gravity? Do enemies continue moving in consistent directions? Does the background scroll when the character moves? These are qualitative evaluations but they reveal whether the model learned predictive structure or just memorized textures.

**Does it generalize to unseen levels?** Procgen procedurally generates levels. Train on levels 0-10000, test on levels 10001-11000. Does prediction quality hold? If yes, the model learned general dynamics, not specific layouts.

These questions produce the analysis that makes your portfolio piece stand out.

---

## Summary of the computational pipeline

```
Raw frame (128×128×3)
    │
    ▼
VQ-VAE Encoder (CNN) ──→ Continuous features (16×16×256)
    │
    ▼
Vector Quantization ──→ Discrete tokens (16×16 indices from codebook of 512)
    │                    [This compressed representation is used for storage/efficiency]
    ▼
VQ-VAE Decoder (CNN) ──→ Reconstructed frame (128×128×3)
    │                    [Verify reconstruction quality during tokenizer training]
    │
    ║ ← The dynamics model works in PIXEL space, not token space
    ║    (Tokens are used only for efficient storage of context)
    ▼
Flow Matching Dynamics Model (U-Net)
    Inputs: noisy interpolation + 4 context frames (channel-concat) + action (AdaGN)
    Output: velocity field prediction (128×128×3)
    │
    ▼
ODE integration (15 Euler steps) ──→ Predicted next frame (128×128×3)
    │
    ▼
Shift context window, repeat
```

Each piece solves exactly one problem. The VQ-VAE solves representation. The flow matching model solves generation. The conditioning mechanism solves action-responsiveness. The noise augmentation solves error accumulation. There are no unnecessary components.
