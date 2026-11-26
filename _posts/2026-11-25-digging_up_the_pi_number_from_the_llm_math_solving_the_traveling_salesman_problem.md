---
layout: post
title: "In which a traveling salesman digs π out of LLM temperature math"
permalink: /stories/2026-11-25-pi-from-simulated-annealing.html
date: 2026-11-25
author: Yours Truly
categories: ML Mathematics Rust
show_homepage: true
description: "The salesman travels, heats up, cools down, bumps into Archimedes"
summary: "The salesman travels, heats up, cools down, bumps into Archimedes"
---

<!-- Include KaTeX CSS and JS files -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body);"></script>

You've probably typed `temperature: 0.7` into an [LLM](https://en.wikipedia.org/wiki/Large_language_model) API payload at some point in the last few years. Or you've heard there's some "temperature" in the AI stack and filed it under "weird stuff people get paid tons of money for".

It isn't that weird, but don't take my word for it. Roughly, and going on vibes: turn the temperature down to `0.0` and the model gets unimaginative and sees red tape everywhere. Turn it up to `1.8` and it switches languages mid-sentence and closes brackets it didn't open yet.

So why does this post also have $$\pi$$ and a traveling salesman in it?

Because that "temperature" has a past. It wandered in from _19th-century [statistical mechanics](https://en.wikipedia.org/wiki/Statistical_mechanics)_ and made itself at home in machine learning.

And once you see where the equation comes from, you start seeing ways to abuse it. Today's abuse: we take the Boltzmann temperature math behind LLM token sampling, hand it to an [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) optimization problem on a unit circle, write a small solver for it, and dig [the number $$\pi$$](https://en.wikipedia.org/wiki/Pi) out of the rubble.

I promise I won't cheat. (We'll get to what "cheating" means in a minute.)

**Contents**

* TOC
{:toc}

---

## So what *is* temperature in an LLM?

At the end of a forward pass, a language model hands you a big vector of raw, unnormalized scores, one per token in its vocabulary, and leaves picking the word to you. People call those scores logits, and we'll write them as $$z_1, z_2, \dots, z_V$$.

To turn the scores into probabilities we can sample from, we run them through [softmax](https://en.wikipedia.org/wiki/Softmax_function), dividing the logits by a positive number $$T$$ on the way in:

$$P_i = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}$$

Look at the numerator for a second: $$e^{z_i / T}$$.

If you took a thermodynamics class in college and have since suppressed the memories, sorry, I'm about to bring one back: the [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution). Take a physical system that can be in a state $$s$$ with energy $$E(s)$$, in a heat bath at temperature $$T$$. You'll find it in that state with probability:

$$P(s) \propto e^{-E(s) / T}$$

The two equations match. An LLM pretends that a token's negative logit ($$-z_i$$) measures its energy. Likely tokens sit at the bottom of a comfortable valley, and the weird ones cling to a cliff way above.

Now turn the knob.

When $$T$$ goes to zero, dividing by a small number blows up the gap between the biggest logit and the rest, and softmax collapses into [`argmax`](https://en.wikipedia.org/wiki/Arg_max). The system gets greedy: it rolls downhill into the nearest well and stays there.

When $$T$$ gets huge, the $$z_i / T$$ all get squashed toward $$0$$, and $$e^0 = 1$$ for everybody. The distribution goes flat. High energy, low energy, the system has stopped caring.

In between, you get a mix. Low-energy states are still preferred, but there's enough thermal jitter to occasionally hop uphill into a worse state, wander into the next valley, and maybe find something better over the ridge.

Metallurgists have known this for a long time. Heat a lump of metal until its atoms are scrambling around, cool it down *slowly*, and the atoms settle into a low-energy crystal lattice. Cool it too fast ([quench](https://en.wikipedia.org/wiki/Quenching) it) and they freeze wherever they happened to be standing, defects and all.

Computer science borrowed the trick and called it [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). And a traveling salesman takes to heat much the way iron does.

---

## A traveling salesman, on a circle

In the [Traveling Salesman Problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem), you get $$N$$ cities and you want the shortest closed loop that visits all of them once and comes back to the start.

In general it's NP-hard. For $$N = 250$$ cities there are $$(250 - 1)! / 2 \approx 6.5 \times 10^{489}$$ distinct tours. If you'd been checking a billion tours per nanosecond since the Big Bang, you wouldn't have made a dent.

But what if we put all $$N$$ cities on a [unit circle](https://en.wikipedia.org/wiki/Unit_circle) (radius $$R = 1$$)?

Then we know what the best tour looks like. The cities lie on the boundary of a convex shape, so the shortest tour doesn't cross itself, and a tour that doesn't cross itself walks around the rim in order, tracing an inscribed convex $$N$$-gon.

As $$N$$ grows, the perimeter $$L$$ of that polygon creeps up on the circumference $$2\pi R = 2\pi$$ from below. Divide the shortest tour length by $$2$$ and out pops $$\pi$$:

$$\hat{\pi} = \frac{L}{2R} = \frac{L}{2}$$

You should get suspicious about now. How am I putting those cities on the circle? If I write `let theta = 2.0 * PI * rng(); (theta.cos(), theta.sin())` and then announce that I "discovered" $$\pi$$, you should close this tab. Using `PI`, `sin` or `cos` to build the circle would be circular reasoning in the most literal sense :D

So we need a no-cheating rule: the city generator and the solver don't get to touch `PI`, `sin`, `cos` or `atan2`. We import `PI` once, at the end of `main()`, to print how close we got.

Which leaves a question: how do you sample uniformly on a circle armed with `+`, `-`, `*`, `/` and `sqrt`? With [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling):

1. Pick a random point $$(x, y)$$ uniformly inside the square $$[-1, 1]^2$$.
2. Compute $$r^2 = x^2 + y^2$$.
3. If $$r^2 > 1$$ (outside the disk) or $$r^2 \approx 0$$ (dead center), throw it away and try again.
4. Otherwise, push it straight out to the rim by dividing by $$r = \sqrt{r^2}$$:

   $$\left(\frac{x}{\sqrt{x^2 + y^2}},\, \frac{y}{\sqrt{x^2 + y^2}}\right)$$

The disk is rotationally symmetric, so projecting uniform points from the disk onto its boundary gives a uniform distribution on the circle. (Tradition demands that I leave the algebra as an exercise.) We didn't need any trig.

*(The fraction of square samples that land inside the disk happens to be $$\pi / 4$$, I know. The solver is handed a bag of $$(x, y)$$ coordinates and has to work out the shortest way to walk between them. The rejected points and that ratio stay behind. So I'm calling the experiment uncontaminated.)*

---

## Cooling a hairball into a circle

Connect our $$N$$ random cities in the order they were generated and you don't get a ring. You get a hairball of chords slicing back and forth across the circle. For $$N = 150$$ that first tour has length $$L \approx 190$$, which makes $$\hat{\pi} \approx 95$$. [Archimedes](https://en.wikipedia.org/wiki/Archimedes) isn't worried yet.

So let's point Boltzmann at the tour:

| Concept | Large Language Models | Simulated Annealing (TSP) |
| :--- | :--- | :--- |
| **State $$s$$** | Next token choice | Current permutation of cities |
| **Energy $$E(s)$$** | Negative logit ($$-z_i$$) | Total tour length $$L$$ |
| **Transition Rule** | Softmax: $$\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$ | [Metropolis](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm): $$P(\text{accept}) = \min\!\left(1,\, e^{-\Delta E / T}\right)$$ |
| **High $$T$$** | Chaotic, flat sampling | Accepts worse tours to escape knots |
| **Low $$T \to 0$$** | Greedy `argmax` | Freezes into the nearest local minimum |

One step of simulated annealing goes like this:

1. Propose a small random change to the tour and compute the energy change $$\Delta E = L_{\text{new}} - L_{\text{old}}$$.
2. If $$\Delta E \le 0$$ (the tour got shorter), take it.
3. If $$\Delta E > 0$$ (the tour got *longer*), draw a uniform random number $$u \in [0, 1)$$ and take it anyway if:

   $$u < e^{-\Delta E / T}$$

4. Nudge the temperature down a little: $$T \leftarrow \alpha T$$, with $$\alpha$$ just under $$1$$ (something like `0.999994`).

### How do you tweak a tour, though?

The obvious random move swaps cities: pick two of them in the tour and exchange their positions.

It works for a while. But when two long stretches of the tour are crisscrossed like a pretzel, swapping two individual cities usually makes things worse before it makes them better, and at low temperature "worse" doesn't get accepted. City swap ends up stuck in a knotted [local minimum](https://en.wikipedia.org/wiki/Maximum_and_minimum).

So we'll use a [2-opt](https://en.wikipedia.org/wiki/2-opt) move instead: pick two indices $$i < j$$ and reverse the slice `tour[i..=j]`.

Why does that work so much better on a circle? Name four cities first: `B = tour[i]` and `C = tour[j]` mark the two ends of the slice, `A` precedes `B` in the tour, and `D` follows `C`. Now picture the edges `A -> B` and `C -> D` crossing in the middle:

```text
  Before (crossed):          After reversing [B .. C]:

    A       C                    A ------- C
     \     /
      \   /          ===>
       \ /
        X
       / \
      /   \
     /     \
    D       B                    D ------- B
```

Reversing everything from `B` to `C` replaces the crossing pair `(A, B)` and `(C, D)` with the uncrossed pair `(A, C)` and `(B, D)`. By the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality), the two diagonals of a convex quadrilateral add up to more than two opposite sides do. So uncrossing a pair of edges with a 2-opt reversal makes the tour shorter.

And we don't have to add up all $$N$$ edges to know $$\Delta E$$. Two edges leave, two edges come in, and the arithmetic takes $$O(1)$$:

$$\Delta E = \big(d(A, C) + d(B, D)\big) - \big(d(A, B) + d(C, D)\big)$$

---

## Melt it yourself

You can read about freezing hairballs, or you can watch one freeze into a clean ring while $$\hat{\pi}$$ slides from `95` down to about `3.140`. I'd watch. (It looks a lot like a [phase transition](https://en.wikipedia.org/wiki/Phase_transition). A physicist might quibble.)

A live simulation runs in your browser below, with the trig-free sampling and the $$O(1)$$ energy update from above. It comes with four presets:

1. **Well-Annealed**: the hot, gassy tour cools down into an uncrossed polygon.
2. **Too Hot** ($$T = 1.5$$): your LLM with `temperature` cranked way up. Magenta flashes mark the *uphill* moves that got accepted, and the tour doesn't settle.
3. **Quenched**: instant freeze ($$T = 10^{-6}$$) with city swap. Like greedy decoding stuck in a loop, it locks into a tangled pretzel and stays there.
4. **300 Cities**: more cities and a slower cooling schedule, for a $$\hat{\pi}$$ around `3.1412`.

Preset 3 cheats a little: it changes the move along with the temperature, because frozen 2-opt doesn't get stuck on a circle. A tangled tour there still has a crossing somewhere, and undoing a crossing shortens the tour, so 2-opt slides downhill all the way to the polygon without help from the heat. City swap does get stuck, the way 2-opt would on an ordinary map where the cities don't line up on a ring.

Once Well-Annealed has settled into a circle, hit **Reheat** to inject a thermal spike ($$T = 1.5$$) and watch the crystal fall apart before it cools back down.

<div style="margin: 2rem 0;">
  <iframe
    id="tsp-pi-demo"
    src="/demos/circular_tsp_simulated_annealing_estimating_pi.html?embed=1"
    title="Simulated annealing on a circle, interactive demo"
    style="width: min(1100px, 96vw); max-width: 100%; height: 920px; border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 2px; display: block; margin: 0 auto; background: transparent;"
    loading="lazy"
  ></iframe>
  <p style="font-size: 0.85em; text-align: right; margin-top: 0.5rem; opacity: 0.8;">
    <a href="/demos/circular_tsp_simulated_annealing_estimating_pi.html" target="_blank" rel="noopener">Open full-screen &nearr;</a>
  </p>
</div>
<script>
  window.addEventListener('message', function (event) {
    if (event.origin !== window.location.origin) return;
    if (event.data && event.data.type === 'tsp-pi-resize' && typeof event.data.height === 'number') {
      var frame = document.getElementById('tsp-pi-demo');
      if (frame) {
        frame.style.height = Math.max(640, Math.ceil(event.data.height) + 8) + 'px';
      }
    }
  });
</script>

---

## The same thing, at full speed

The browser shows the physics at 60 frames per second. For 2.5 million [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) steps we want something faster, so we'll leave the browser and compile it.

The program doesn't need any dependencies, so we won't even create a `Cargo.toml`. It fits in one file, `tsp_pi.rs`: a small [`xorshift64`](https://en.wikipedia.org/wiki/Xorshift) PRNG, the trig-free circle sampler, and the 2-opt Boltzmann loop.

```rust
use std::f64::consts::PI; // Used ONLY at the very end to check our error

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform float in [0.0, 1.0)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, bound)
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn dist(self, other: Point) -> f64 {
        // Plain sqrt rather than hypot: the coordinates stay within [-1, 1],
        // so the squares can't overflow, and sqrt costs less in a hot loop
        let (dx, dy) = (self.x - other.x, self.y - other.y);
        (dx * dx + dy * dy).sqrt()
    }
}

/// Sample `n` uniformly random points on the unit circle (R = 1)
/// without ever touching PI, sin, or cos.
fn sample_unit_circle(n: usize, rng: &mut Rng) -> Vec<Point> {
    let mut pts = Vec::with_capacity(n);
    while pts.len() < n {
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = rng.next_f64() * 2.0 - 1.0;
        let r2 = x * x + y * y;
        if r2 > 1e-12 && r2 <= 1.0 {
            let r = r2.sqrt();
            pts.push(Point { x: x / r, y: y / r });
        }
    }
    pts
}

fn tour_length(cities: &[Point], tour: &[usize]) -> f64 {
    let n = tour.len();
    (0..n)
        .map(|i| cities[tour[i]].dist(cities[tour[(i + 1) % n]]))
        .sum()
}

fn solve_tsp_annealing(
    cities: &[Point],
    t_start: f64,
    t_end: f64,
    steps: usize,
    rng: &mut Rng,
) -> Vec<usize> {
    let n = cities.len();
    let mut tour: Vec<usize> = (0..n).collect();
    let mut current_len = tour_length(cities, &tour);

    let mut best_tour = tour.clone();
    let mut best_len = current_len;

    // Exponential cooling factor: T_k = t_start * alpha^k
    let alpha = (t_end / t_start).powf(1.0 / steps as f64);
    let mut temp = t_start;

    for step in 0..steps {
        if step % (steps / 5) == 0 {
            println!(
                "step {:>8} | T = {:<10.6} | length = {:<10.5} | pi_est = {:.6}",
                step,
                temp,
                current_len,
                current_len / 2.0
            );
        }

        let mut i = rng.next_usize(n);
        let mut j = rng.next_usize(n);
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }

        // Skip proposals that do nothing: a single city, or the whole tour
        // (reversing all of it only flips the direction)
        if i != j && !(i == 0 && j == n - 1) {
            // Reversing tour[i..=j] trades edges (a, b), (c, d) for (a, c), (b, d)
            let a = cities[tour[(i + n - 1) % n]];
            let b = cities[tour[i]];
            let c = cities[tour[j]];
            let d = cities[tour[(j + 1) % n]];

            let delta = (a.dist(c) + b.dist(d)) - (a.dist(b) + c.dist(d));

            // Shorter: take it. Longer: take it with probability e^(-delta / T)
            if delta <= 0.0 || rng.next_f64() < (-delta / temp).exp() {
                tour[i..=j].reverse();
                current_len += delta;

                if current_len < best_len {
                    best_len = current_len;
                    best_tour.clone_from(&tour);
                }
            }
        }

        temp *= alpha; // cool down whether or not anything moved
    }

    best_tour
}

fn main() {
    let n = 250;
    let steps = 2_500_000;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    let cities = sample_unit_circle(n, &mut rng);
    let initial_len = tour_length(&cities, &(0..n).collect::<Vec<_>>());

    println!("Cities on unit circle: {}", n);
    println!("Initial random tour length: {:.4}\n", initial_len);

    let best_tour = solve_tsp_annealing(&cities, 2.0, 1e-6, steps, &mut rng);

    // Recompute exact length from scratch to avoid floating-point drift
    let final_len = tour_length(&cities, &best_tour);
    let pi_est = final_len / 2.0;

    println!("\nFinal tour length : {:.8}", final_len);
    println!("Estimated Pi      : {:.8}", pi_est);
    println!("Actual Pi         : {:.8}", PI);
    println!("Absolute error    : {:.8}", (PI - pi_est).abs());
}
```

Compile it with optimizations (benchmarking a debug build tends to end in sadness) and run it:

```bash
rustc -O tsp_pi.rs && ./tsp_pi
```

About 40 milliseconds later, I get this:

```text
Cities on unit circle: 250
Initial random tour length: 327.8122

step        0 | T = 2.000000   | length = 327.81220  | pi_est = 163.906099
step   500000 | T = 0.109856   | length = 34.15688   | pi_est = 17.078442
step  1000000 | T = 0.006034   | length = 12.18384   | pi_est = 6.091918
step  1500000 | T = 0.000331   | length = 12.02600   | pi_est = 6.012998
step  2000000 | T = 0.000018   | length = 6.28175    | pi_est = 3.140874

Final tour length : 6.28174788
Estimated Pi      : 3.14087394
Actual Pi         : 3.14159265
Absolute error    : 0.00071871
```

Follow the `pi_est` column down: `163.9`, `17.1`, and then... `6.09`? And `6.01`?

For half a million steps, our salesman believed in a $$\pi$$ of 6.

The tour folded. Instead of going around the circle once, it runs most of the way around and then doubles back, so it covers the rim two times and comes out roughly two times as long as it should. It isn't stuck there, exactly. As far as I can tell, the moves that would unfold it only shave a hair off the length, and while the temperature exceeds that hair, the tour shrugs and keeps wandering. Once $$T$$ gets low enough, the fold unzips and we arrive at `3.140874`.

I tried 400 other seeds, and about 45% of them take the same detour. They all got out before the end.

(Your middle rows may look different. The run is seeded, but `exp` and `powf` aren't bit-identical across platforms. The final answer should match.)

One bit of numerical hygiene at the end of `main()`: inside the loop we keep `current_len += delta` up to date in $$O(1)$$, and over millions of additions that can pick up a little [IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) rounding drift. So at the end we recompute `tour_length(&cities, &best_tour)` from scratch, and the final $$\hat{\pi}$$ uses the real perimeter of that polygon.

---

## Oh no, why did it stop at `3.1409`?

Hold on. At step 2,000,000 it reads `3.140874`, and the final answer agrees: `3.14087394`. It stopped moving, and the fourth digit is still wrong. Did the annealing get stuck after all?

The annealing did its job. It found the best tour there is: one that doesn't cross itself anywhere, the shortest closed loop through those 250 cities.

Geometry accounts for the leftover `0.0007`. A chord is shorter than the arc it cuts off, so even a perfect TSP solver comes up short of $$\pi$$ on a circle.

We can even estimate by how much. A chord spanning a small central angle $$\theta$$ has length $$2\sin(\theta / 2)$$, and the arc has length $$\theta$$. Expanding the [Taylor series](https://en.wikipedia.org/wiki/Taylor_series):

$$2\sin\left(\frac{\theta}{2}\right) \approx \theta - \frac{\theta^3}{24}$$

With $$N$$ evenly spaced points, the gaps would all be $$\theta = 2\pi / N$$ and the shortfall would be $$\frac{\pi^3}{6N^2}$$.

Our points are random, though, so some gaps are wider than average and some are narrower. The error goes like the *cube* of the gap, which means the wide ones hurt a lot more than the narrow ones help. Working through the [order statistics](https://en.wikipedia.org/wiki/Order_statistic) of uniform spacings on a circle ($$\mathbb{E}[\theta^3] \approx 6\,(2\pi / N)^3$$), the expected shortfall comes out $$6\times$$ larger than for a regular polygon:

$$\mathbb{E}[\hat{\pi}] \approx \pi - \frac{\pi^3}{N^2}$$

Plug in $$N = 250$$:

$$\frac{\pi^3}{250^2} \approx \frac{31.0063}{62500} \approx 0.000496$$

Our run is off by `0.000719`. That falls in the right ballpark, but 45% above the prediction, and I can't blame a bug for it either. $$\pi^3/N^2$$ predicts the *average* over city sets, and a single set wobbles around it by about 20%. I ran 200,000 random sets to check: the mean shortfall came out at `0.000490`, with a standard deviation of `0.000096`. So theory and Monte Carlo agree on the average, and our seed was unlucky, somewhere past the 95th percentile. The cube again: a few wide gaps will do that.

(The Monte Carlo also caught me rounding. Keep the spacings exact and the average works out to $$\frac{\pi^3}{(N+1)(N+2)}$$, or `0.000490` for $$N = 250$$. $$\pi^3/N^2$$ approximates that for large $$N$$.)

Want two more digits of $$\pi$$? Multiply $$N$$ by $$10$$ (to $$2{,}500$$ cities). 2-opt has $$\binom{N}{2} = O(N^2)$$ pairs of edges to try, though, so you'll want roughly $$100\times$$ more annealing steps too.

---

## Second act, for the most demanding reader: Pi from the heat

If you count yourself among the most demanding readers, you've been sitting on two objections since the first listing. They're both fair.

The first one: the circle did it. We read our $$\pi$$ off the perimeter of a polygon inscribed in a circle. Boltzmann found the polygon for us, but the digits belong to Archimedes.

The second one: the temperature had an easy job. The Quenched preset gave that away: with 2-opt on a circle the tour doesn't have traps to climb out of, so a [greedy](https://en.wikipedia.org/wiki/Greedy_algorithm) solver that only accepts improvements gets there too. You can try it: change the call to `solve_tsp_annealing(&cities, 1e-9, 1e-9, steps, &mut rng)`, and the progress line at step 500,000 already shows the final tour length. It even skips the detour through $$\pi \approx 6$$. The heat earns its keep with clumsier moves, like the city swap in the demo.

So the second act plays by stricter rules. The temperature has to do the work, and by the end the circle has to go.

### Don't cool it all the way

Take the frozen tour and, instead of cooling it to zero, hold it at a small fixed temperature $$T$$. It won't hold still. Once in a while two neighboring cities trade places, the tour gets slightly longer, and some time later the mistake gets undone. We'll measure the *average* excess length $$\langle L \rangle - L_0$$ over the frozen length $$L_0$$.

A greedy solver wouldn't budge from the frozen tour, so its excess would be zero all day. Whatever we measure here, the temperature made it.

### One pair of cities behaves like a two-token softmax

Let $$a, b, c, d$$ be four consecutive cities on the rim, and let $$g$$ be the gap between $$b$$ and $$c$$, measured along the rim. A 2-opt reversal of the slice $$[b, c]$$ visits them as $$a, c, b, d$$ instead, and our $$O(1)$$ formula prices it:

$$\Delta E = d(a, c) + d(b, d) - d(a, b) - d(c, d) \approx 2g$$

The tour now crosses that gap three times instead of once: forward, back, and forward again.

So this pair has two states: "in order" with energy $$0$$, and "swapped" with energy $$2g$$. Boltzmann tells us how often we'll catch it swapped:

$$P(\text{swapped}) = \frac{e^{-2g/T}}{e^{0} + e^{-2g/T}} = \frac{1}{1 + e^{2g/T}}$$

Look at the middle expression. You've seen it before: the softmax from the top of this post produces it for a vocabulary of two tokens with logits $$0$$ and $$-2g$$, sampled at temperature $$T$$. Machine learning people call the right-hand side a [sigmoid](https://en.wikipedia.org/wiki/Logistic_function). On average, this pair adds $$2g \cdot P(\text{swapped})$$ to the tour.

### Add up all the pairs

Gaps much wider than $$T$$ are exponentially suppressed, so the tiny ones do most of the work. How many of those are there? The gap from a city to its next neighbor is longer than $$g$$ when the other $$N - 1$$ cities all miss an arc of length $$g$$, and that happens with probability $$(1 - g/2\pi)^{N-1}$$. So tiny gaps are spread evenly, with density $$(N-1)/2\pi$$ per unit of length, and there are $$N$$ gaps in total:

$$\langle L \rangle - L_0 \approx N \cdot \frac{N-1}{2\pi} \int_0^\infty \frac{2g}{1 + e^{2g/T}}\, dg = \frac{N(N-1)}{2\pi} \cdot \frac{T^2}{2} \int_0^\infty \frac{x}{1 + e^{x}}\, dx$$

Keep an eye on that last integral. Write $$\frac{1}{1 + e^{x}} = e^{-x} - e^{-2x} + e^{-3x} - \dots$$, integrate term by term, and out comes

$$\int_0^\infty \frac{x}{1 + e^{x}}\, dx = 1 - \frac{1}{4} + \frac{1}{9} - \frac{1}{16} + \dots = \frac{\pi^2}{12}$$

Mathematicians file that sum under the [alternating cousin](https://en.wikipedia.org/wiki/Dirichlet_eta_function) of the [Basel problem](https://en.wikipedia.org/wiki/Basel_problem). We didn't put a circle into that integral. We put in a sigmoid times $$x$$. Put it all together:

$$\langle L \rangle - L_0 \approx \frac{\pi\, N(N-1)\, T^2}{48} \quad\Longrightarrow\quad \hat{\pi} = \frac{48\, \big(\langle L \rangle - L_0\big)}{N(N-1)\, T^2}$$

The softmax handed us $$\pi^2$$, and the circumference $$2\pi$$ in the gap density took one $$\pi$$ back. Hold that thought.

Some fine print before we run anything.

Call it a cold law: it wants $$T$$ well below the typical gap $$2\pi/N$$. Reversing a longer run of cities costs two times its full span, so it takes two or more tiny gaps in a row, and that shows up at order $$T^3$$. The same order catches a shortcut we took: we counted gaps as if they were spread evenly at all sizes, and they thin out as they get wider.

And the randomness supplies the signal. On a regular polygon a swap costs the same $$4\pi/N$$ wherever you try it, so near-ties don't come up and the excess is exponentially small. The random spacing cost us a factor of $$6$$ in the first act, and it gives us something to measure in the second.

### The code

Same rules as before: one file, and `PI`, `sin`, `cos` and `atan2` stay away from the experiment. A few things changed.

We start frozen. The first act already showed where the solver ends up, so the cities are sorted around the rim (with a trig-free key) and the clock starts there.

We pin one city. It occupies both ends of the array, which turns the closed tour into a path with fixed ends. Reversing an interior slice still covers all the 2-opt moves. That includes trading the pinned city with a neighbor: on a closed loop, reversing the rest of the cities draws the same tour.

And we average over tens of thousands of city sets, because a single set gambles on its handful of nearly tied pairs. The program reports a [standard error](https://en.wikipedia.org/wiki/Standard_error) so you can see how much to believe.

Save this as `heat_pi.rs`:

```rust
use std::f64::consts::PI; // Used ONLY on the last line, to grade the answers

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn dist(self, other: Point) -> f64 {
        let (dx, dy) = (self.x - other.x, self.y - other.y);
        (dx * dx + dy * dy).sqrt()
    }
}

/// `n` cities on the unit circle, already in their frozen order around the rim.
/// One city is pinned at (1, 0) and sits at both ends of the array, which turns
/// the closed tour into a path with fixed ends.
fn circle(n: usize, rng: &mut Rng) -> Vec<Point> {
    let pin = Point { x: 1.0, y: 0.0 };
    let mut pts = vec![pin];
    while pts.len() < n {
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = rng.next_f64() * 2.0 - 1.0;
        let r2 = x * x + y * y;
        if r2 > 1e-12 && r2 <= 1.0 {
            let r = r2.sqrt();
            pts.push(Point { x: x / r, y: y / r });
        }
    }
    // Still no trig: this key grows as you walk counterclockwise from (1, 0).
    let key = |p: &Point| {
        let s = p.x.abs() + p.y.abs();
        if p.y >= 0.0 { 1.0 - p.x / s } else { 3.0 + p.x / s }
    };
    pts.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    pts.push(pin);
    pts
}

/// `n` points on the straight segment from 0 to 1, in order, with the path
/// pinned at both ends. Nothing round in sight.
fn segment(n: usize, rng: &mut Rng) -> Vec<Point> {
    let mut pts: Vec<Point> = (0..n).map(|_| Point { x: rng.next_f64(), y: 0.0 }).collect();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    pts.insert(0, Point { x: 0.0, y: 0.0 });
    pts.push(Point { x: 1.0, y: 0.0 });
    pts
}

/// Hold a tour at temperature `t`; return its average excess length over the frozen tour.
fn mean_excess(mut tour: Vec<Point>, t: f64, steps: usize, rng: &mut Rng) -> f64 {
    let m = tour.len() - 2; // movable cities; the two ends don't move
    let mut excess = 0.0; // we start frozen
    let mut sum = 0.0;
    for step in 0..steps {
        let i = 1 + rng.next_usize(m);
        let j = 1 + rng.next_usize(m);
        if i != j {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let (a, b, c, d) = (tour[i - 1], tour[i], tour[j], tour[j + 1]);
            let delta = (a.dist(c) + b.dist(d)) - (a.dist(b) + c.dist(d));
            // Same Boltzmann / Metropolis rule as before, except T never changes
            if delta <= 0.0 || rng.next_f64() < (-delta / t).exp() {
                tour[i..=j].reverse();
                excess += delta;
            }
        }
        if step >= steps / 2 {
            sum += excess; // the first half is warm-up
        }
    }
    sum / (steps - steps / 2) as f64
}

/// Average over many independent city sets. Returns (mean, standard error).
fn experiment(
    make: fn(usize, &mut Rng) -> Vec<Point>,
    n: usize,
    t: f64,
    sets: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let steps = 6 * n * n; // a given pair of neighbors comes up about once per n^2 / 2 steps
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for _ in 0..sets {
        let e = mean_excess(make(n, rng), t, steps, rng);
        sum += e;
        sum_sq += e * e;
    }
    let mean = sum / sets as f64;
    let var = sum_sq / sets as f64 - mean * mean;
    (mean, (var / sets as f64).sqrt())
}

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn main() {
    let n = 100;
    let nn = (n * (n - 1)) as f64;

    println!("Circle, {} cities:  pi = 48 * excess / (N (N-1) T^2)", n);
    let mut rng = Rng(SEED);
    for &(t, sets) in &[(0.01, 30_000), (0.025, 5_000), (0.05, 5_000)] {
        let (e, se) = experiment(circle, n, t, sets, &mut rng);
        let k = 48.0 / (nn * t * t);
        println!("  T = {:<6} | excess = {:.6} | pi_est = {:.4} +- {:.4}", t, e, k * e, k * se);
    }

    println!("Segment, {} points: pi^2 = 24 * excess / (N (N-1) T^2)", n);
    let mut rng = Rng(SEED);
    for &(t, sets) in &[(0.0016, 30_000)] {
        let (e, se) = experiment(segment, n, t, sets, &mut rng);
        let k = 24.0 / (nn * t * t);
        let p2 = k * e;
        println!("  T = {:<6} | excess = {:.6} | pi^2_est = {:.4} +- {:.4}", t, e, p2, k * se);
        println!("  square root: pi_est = {:.4} +- {:.4}", p2.sqrt(), 0.5 * k * se / p2.sqrt());
    }

    println!("Actual: pi = {:.4}, pi^2 = {:.4}", PI, PI * PI);
}
```

```bash
rustc -O heat_pi.rs && ./heat_pi
```

This one takes a bit longer than 40 milliseconds. About 50 seconds and 4.2 billion Metropolis steps later:

```text
Circle, 100 cities:  pi = 48 * excess / (N (N-1) T^2)
  T = 0.01   | excess = 0.064633 | pi_est = 3.1337 +- 0.0059
  T = 0.025  | excess = 0.400931 | pi_est = 3.1103 +- 0.0090
  T = 0.05   | excess = 1.533594 | pi_est = 2.9742 +- 0.0057
Segment, 100 points: pi^2 = 24 * excess / (N (N-1) T^2)
  T = 0.0016 | excess = 0.010386 | pi^2_est = 9.8352 +- 0.0187
  square root: pi_est = 3.1361 +- 0.0030
Actual: pi = 3.1416, pi^2 = 9.8696
```

At $$T = 0.01$$, or $$1/N$$, about a sixth of the typical gap, the heat of the tour says $$\hat{\pi} = 3.134 \pm 0.006$$. This time the number reflects how often neighboring cities get caught out of order, and the perimeter stays out of it.

The warmer rows show the fine print at work: $$3.110$$, then $$2.974$$. The two corrections pull in opposite directions. Longer flipped runs add length, the thinning supply of wider gaps takes some away, and on balance the estimate sags, the way a cold law should. Colder is cleaner but also quieter, because the signal shrinks like $$T^2$$ and it takes more city sets to hear it. If you have a quarter of an hour to spare, swap the circle rows for a single `(0.006, 400_000)` and the segment row for `(0.001, 400_000)`. The circle then reads $$3.1401 \pm 0.0021$$ and the segment $$3.1389 \pm 0.0010$$. Both get closer, and the error bars have shrunk enough that the last sliver of cold-law bias shows.

A single city set scatters by about 30% around these values. Multiply the standard error by $$\sqrt{30000}$$ and see for yourself.

### Now take the circle away

Remember the $$2\pi$$ that took one of our two $$\pi$$s back? I can't find another spot where the circle got in. So we delete the circle.

Scatter $$N$$ points on a straight segment of length $$1$$ and ask for the shortest path from one end to the other through all of them. The sorted order wins, and it has length $$1$$. Swapping two neighbors costs $$2g$$ on the nose this time, and we can drop the approximation. There are $$N - 1$$ neighboring pairs, and a gap is longer than $$g$$ with probability $$(1 - g)^N$$ (bend the segment into a ring and let one extra point mark the seam), so tiny gaps have density $$N$$. The same sum gives

$$\langle L \rangle - 1 \approx (N-1) \cdot N \cdot \frac{T^2}{2} \cdot \frac{\pi^2}{12} = \frac{\pi^2\, N(N-1)\, T^2}{24}$$

This time the $$\pi^2$$ gets to keep both of its $$\pi$$s. The `Segment` block of the output shows it: $$\hat{\pi}^2 = 9.835 \pm 0.019$$, so $$\hat{\pi} = 3.136 \pm 0.003$$.

That experiment got by without a circle, a rim, a polygon or a single call to trig. We jiggled a list of sorted numbers with the Boltzmann factor that also lives inside softmax, and $$\pi$$ fell out anyway.

### What the demanding reader takes home

An LLM person would read it like this. Take two tokens whose logits differ by $$\Delta$$. Sampling at temperature $$T$$ instead of taking the `argmax` picks the worse one with probability $$1/(1 + e^{\Delta/T})$$, and gives up $$\Delta$$ of logit when it does. Add that loss up over all the near-ties, $$\Delta$$ from $$0$$ to $$\infty$$, and you get $$\pi^2 T^2 / 12$$. We just dug that number up, and it had been in the sampling math all along.

A physicist would read it differently. Differentiate the excess with respect to $$T$$ and you get a [heat capacity](https://en.wikipedia.org/wiki/Heat_capacity) that grows linearly with temperature. Physicists use the same flat spread of two-level systems as their standard (if still debated) picture of why the heat capacity of [cold glasses](https://en.wikipedia.org/wiki/Amorphous_solid#Universal_low-temperature_properties_of_amorphous_solids) grows linearly with $$T$$, and textbooks list the integral as a [Fermi–Dirac integral](https://en.wikipedia.org/wiki/Complete_Fermi%E2%80%93Dirac_integral). Our tour behaves like a very small glass.

And then the bill arrives: 4.2 billion steps for a shaky `3.13`, where the first act needed 2.5 million for a solid `3.14`. As a way to compute $$\pi$$, the second act is about a thousand times worse than the first, which was already terrible. As a way to make the title true, though, it goes further than the rest of this post.

---

## What did we just do?

The first act pointed the Boltzmann math behind LLM token sampling at an NP-hard problem and spent two and a half million Monte Carlo steps on three digits of $$\pi$$. Along the way we reinvented Archimedes' 250 BCE [inscribed-polygon method](https://en.wikipedia.org/wiki/Measurement_of_a_Circle) by thermodynamic brute force.

The second act stopped cooling, threw the circle away, and found $$\pi^2/12$$ inside a two-token softmax. It took 4.2 billion steps and came back with fewer digits. [Chudnovsky's algorithm](https://en.wikipedia.org/wiki/Chudnovsky_algorithm) gets about fourteen out of a single term of its series.

I'd do it again anyway. I can't say any of it helps me pick a `temperature`, but these days when I reach for that knob I picture a Boltzmann factor deciding whether to climb a hill, and a hairball of chords slowly cooling into a circle.

---

A postscript on the word _optimization_, since in LLM talk it mostly means training. The TSP belongs to the same family, in its brutally hard wing: a tour doesn't have a slope to follow, so we could only propose a change and let the temperature decide. About the simplest member of that family, on the other hand, probably crossed your path in college. Remember fitting a straight line through a cloud of 2D points with [least squares](https://en.wikipedia.org/wiki/Least_squares)? The line can't pass through all the points, so you settle for the one that misses them by the least in total. Training an LLM works the same way. The next tokens in the training text stand in for the points, trillions of them. The model has billions of knobs instead of two, and it draws a curve rather than a line, so the answer doesn't come with a formula. You roll downhill on the total miss instead. That procedure goes by the name of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and you get the trained model wherever you stop. The temperature in this post shows up afterwards, when that model picks its next token.
