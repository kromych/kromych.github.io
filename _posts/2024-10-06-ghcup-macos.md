---
layout: post
title:  "How can one install Haskell on macOS?"
permalink: /stories/2024-10-06-ghcup-macos.html
date:   2024-10-06 16:32:00 -0800
author: Yours Truly
categories: Haskell macOS
show_homepage: true
summary: Haskell, Stokes' theorem, functional programming, macOS - quite a medley!
---

<style type="text/css">
    kbd {
        background-color: #1e1e1e;
        color: #00ff00;
        border: 1px solid #00ff00;
        border-radius: 3px;
        box-shadow: inset 0 -1px 0 #00ff00;
        padding: 2px 6px;
        font-size: 0.85em;
        font-family: "Courier New", Courier, monospace;
        text-shadow: 0 0 5px #00ff00;
    }
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

## Why not about TempleOS, patching KDE under FreeBSD, or what have you

[Haskell](https://www.haskell.org/) is an awesome functional programming language,
and this post being an inaugural one for this blog could not have possibly been
about anything else.

Functional languages naturally give a feeling of being as generic as possible providing
all the facilities to be as close to mathematics as one's code could be. The **Stokes' theorem**
works for a whole lot of differential forms and manifolds and no sightings of `if`
statements, loops, etc at all:

$$
\int_{\partial \Omega} \omega = \int_{\Omega} d\omega
$$

And that helps designing microwave ovens and antennas, true story!

If only there was a way to deal with complexity like that in software engineering:
create good abstractions and the rules of combining them to have less code! Wait, that
is what functional programming is about!

An example of some Haskell [might be the quick sort](https://wiki.haskell.org/Introduction#Quicksort_in_Haskell):

```haskell
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (p:xs) = (quicksort lesser) ++ [p] ++ (quicksort greater)
    where
        lesser  = filter (< p) xs
        greater = filter (>= p) xs
```

Woah, that is way shorter than in C! Truth be told, C allows cutting some
corners to make things faster. On another hand, developing and supporting
*6* lines of code does have its merits. Not that you'll have to write quick
sort yet less code and being generic are always going to be there.

Enough preaching though, we were going to install something, weren't we?

## How one can install Haskell on macOS!

I've tried various approaches and the following appears to provide the
well-organized setup with just one package installed via `Homebrew`, and
everything else goes under `~/.ghcup` and `~/.cabal`.

1. `brew install ghcup`
2. `ghcup tui` for a nice TUI-driven selection of the tools. In the list,
   use the arrow keys <kbd>↑</kbd> <kbd>↓</kbd> to select the component
   and <kbd>i</kbd> to install. `ghc` is the compiler, `hls` is the language
   server provider, and `cabal` is the package and project manager.
3. `export PATH="$HOME/.cabal/bin:$HOME/.ghcup/bin:$PATH"` goes into
   your shell's rc file.
4. Get versions with `ghcup list` and set defaults with `ghcup set ghc 9.10.1`,
   `ghcup set cabal 3.12.1.0`, etc.

As a side note, `rustup` and `cargo` in the Rust ecosystem have been influenced
by `ghcup` and `cabal` from the looks of it. Good for Rust! Oh, and the Haskell
package repository aka [Hackage](https://hackage.haskell.org/) appears to have given
some ideas to [crates.io](https://crates.io).

## My favorite books on Haskell

* [Thinking with types](https://leanpub.com/thinking-with-types)
* [Get Programming with Haskell](https://www.amazon.com/Get-Programming-Haskell-Will-Kurt/dp/1617293768)
* [Learn Physics with Functional Programming: A Hands-on Guide to Exploring Physics with Haskell](https://www.amazon.com/Learn-Physics-Functional-Programming-Hands/dp/1718501668)
* [Algorithm Design with Haskell](https://www.amazon.com/Algorithm-Design-Haskell-Richard-Bird/dp/1108491618/)
* [Haskell for FPGA Hardware Design](https://gergo.erdi.hu/retroclash/)

If you're just starting out, then give a read to
[Learn You a Haskell for Great Good A Beginners Guide](https://learnyouahaskell.com/)

## Am I using that?

At home, yes, I am. It's fun to figure out the solution for dynamic programming
problems :) At work, no, I am not. The work stuff requires top performance and the
ability to program hardware, and Haskell might not be the best tool for
any of that.

All told, hoping to talk about some hardcore system stuff on the blog soon!

## Thank you, Yours Truly :)
