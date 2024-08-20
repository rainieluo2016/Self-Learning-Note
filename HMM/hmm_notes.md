# SLP3 Chapter A

[source](https://web.stanford.edu/~jurafsky/slp3/A.pdf)

## Overview

HMM shoulds be characterized by 3 foundamental problems:

+ **Problem 1 (Likehood):**  Given an HMM $\lambda = (A,B)$ and an observation sequence $O$, detemrine the likelihood $P(O|\lambda)$

+ **Problem 2 (Decoding):** Given and observation sequence $O$ and an HMM $\lambda = (A,B)$, discover the best hidden state sequence $Q$

+ **Problem 3 (Learning)**: Given and observtion sequence $O$ and a set of state in HMM, learn the HMM parameters $A,B$

