---
layout: post
title:  "What is that neural network actually doing?"
permalink: /stories/2026-07-26-mechanistic-interpretability.html
date:   2026-07-26 10:15:00 -0800
author: Yours Truly
categories: ML Mathematics
show_homepage: true
summary: Mechanistic interpretability (linear algebra, group theory and a PDE walk into a neural network).
---

<style type="text/css">
    .post-categories {
        text-align: right;
        margin-top: 10px;
        font-size: 0.9rem;
        font-family: "Courier New", Courier, monospace;
        color: #00ff00;
    }
    .post-categories a {
        color: #00ff00;
        text-decoration: none;
        margin-left: 10px;
    }
    .post-categories a:hover {
        color: #33ff33;
        text-decoration: underline;
    }
    table.logits {
        font-family: "Courier New", Courier, monospace;
        border-collapse: collapse;
        margin: 1em 0;
    }
    table.logits th, table.logits td {
        border: 1px solid #00ff00;
        padding: 4px 12px;
        text-align: center;
    }
</style>

<head>
  <!-- Include KaTeX CSS and JS files -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body);"></script>
</head>

<div class="post-categories">
  {% for category in page.categories %}
    <a href="/categories/{{ category }}">{{ category }}</a>
  {% endfor %}
</div>

## Why would one read the weights?

A trained network is a pile of floats that happens to work. _Mechanistic interpretability_ is
the attempt to read that as a program. That looks as instead of

> _"the model pays attention to token five"_

saying

> _"these two heads implement a lookup. Here is the algorithm, and here is what breaks when I delete it"_ .

That is reminiscent of reverse engineering. The binary is written by gradient descent, there are no
symbols, and the calling convention is linear algebra :D

## Features are directions, and there are far too many of them

The _residual stream_ of a transformer is a vector space $$\mathbb{R}^d$$ that every layer
reads from and adds to. The working hypothesis is that a concept is a direction within it,
and that concepts compose by _vector addition_.

A serious model tracks far more than $$d$$ concepts though, hence they get packed.

_Johnson-Lindenstrauss_ claims that $$\mathbb{R}^d$$ holds $$\exp(O(\varepsilon^2 d))$$
unit vectors that are pairwise almost orthogonal, so $$|\langle u, v \rangle| \le \varepsilon$$ -
exponentially many. So if features are sparse (== most are off for most inputs), the network can
store them in an overcomplete dictionary and eat the interference. That is superposition, and it
shows that a single neuron will fire for academic citations, English dialogue, HTTP requests and
Korean text all at once. Unpacking it is dictionary learning where you fit a wide, sparse autoencoder
to the activations and hope the atoms come out one concept at a time.

The residual stream has _no privileged basis_ speaking more of the linear algebra.

If you rotate it by any orthogonal $$Q$$, and push $$Q^{\top}$$ into the next weight matrix, then the network
computes the identical function. So "coordinate 1337 of the residual stream" doesn't mean anything. The MLP
hidden layer is privileged due to $$\mathrm{ReLU}$$ acting coordinate-wise, and does not commute
with rotations. Symmetry decides where the coordinates mean anything.

## The network that rediscovered the DFT

If you train a small transformer on $$a + b \bmod p$$ and read the weights afterwards then the
embedding rows are sines and cosines at a handful of frequencies. The thing implements a discrete
Fourier transform out of the blue :) The phenomenon is called _grokking_.

With a bit of group theory jargon, here is how that goes. The irreducible representations of the
cyclic group $$\mathbb{Z}/p\mathbb{Z}$$ are exactly the characters $$\chi_k(a) = e^{2\pi i k a/p}$$,
and they are the basis in which group convolution becomes pointwise multiplication. "Add mod $$p$$"
_is_ convolution on that group. Here, the gradient descent found the change of basis.

### Trace it by hand

Let $$p = 5$$, $$\omega_k = 2\pi k/5$$, and ask for $$3 + 4$$.

1. Embed each input onto the unit circle, $$a \mapsto (\cos \omega_k a,\ \sin \omega_k a)$$.
At $$k = 1$$ that is $$3 \mapsto (-0.809, -0.588)$$ and $$4 \mapsto (0.309, -0.951)$$.

2. Multiply the coordinates pairwise. That is what an attention head followed by a
roughly-quadratic nonlinearity hands us. Collecting terms gives:

$$
\cos \omega a \cos \omega b - \sin \omega a \sin \omega b = \cos \omega (a + b)
$$

Numerically, $$(-0.809)(0.309) - (-0.588)(-0.951) = -0.809$$, which is $$\cos 7\omega_1$$. The sum
now lives in a phase, and a phase is mod $$p$$ for free. Hence, no carry, comparisons, and branching.

3. Unembed into one logit per candidate answer $$c$$:

$$
\mathrm{logit}(c) = \sum_k \cos \big( \omega_k (a + b - c) \big)
$$

4. Read it off. Characters are orthogonal so their sum $$\sum_{k=1}^{p-1} \cos(2\pi k m / p)$$
is $$p - 1$$ when $$m \equiv 0$$ and $$-1$$ otherwise. Two frequencies suffice here, since $$\cos$$
is even and so $$k \in \{1, 2\}$$ covers all four nonzero ones:

<table class="logits">
<tr><th>c</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>
<tr><th>logit</th><td>-0.5</td><td>-0.5</td><td><b>2.0</b></td><td>-0.5</td><td>-0.5</td></tr>
</table>

A discrete delta at $$c = 2 = 7 \bmod 5$$. This is exact, shows what people found by staring at a trained
network, then confirming by ablating individual frequencies and watching accuracy fall over.

## Layers are time

A residual block is $$x_{l+1} = x_l + f_l(x_l)$$, which is one forward Euler step of
$$\dot{x} = f(x, t)$$. One might imagine a deep network as a discretized flow and depth as time, so
interpretability gets to ask what this flow transports, and where it "mixes".

Training is a flow too. Gradient descent integrates $$\dot{\theta} = -\nabla L(\theta)$$; put the
minibatch noise back in and the distribution of parameters obeys Fokker-Planck:

$$
\partial_t \rho = \nabla \cdot (\rho \nabla L) + \tfrac{\beta}{2} \Delta \rho
$$

with the Gibbs steady state $$\rho_\infty \propto e^{-2L/\beta}$$. If the width is infinity, then
density of neurons obeys a continuity equation (!), namely $$\partial_t \rho + \nabla \cdot (\rho v) = 0$$.
Looks like a transport.

Perhaps grokking is a phase transition in that flow, and maybe because of that it looks so abrupt from the loss curve.

It is interesting to think how a Green's function fits into the picture.

## Am I using that?

Not at work (yet?). The things I debug there still have source, and I can attach a debugger. At home,
sure. The modular addition network fits on one screen, trains in a couple of minutes on a laptop,
and you can print the embedding matrix and see the sine waves sitting in it.

Much the same feeling as the first time a hex dump turned into a file format.

## What I have been reading

* Olah et al., [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)
* Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
* Elhage et al., [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
* Nanda et al., [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)
* Chughtai, Chan, Nanda, [A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations](https://arxiv.org/abs/2302.03025)
* Templeton et al., [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

## Thank you, Yours Truly :)
