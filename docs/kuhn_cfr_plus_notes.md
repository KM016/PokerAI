# CFR+ — What Changes from Vanilla CFR?

*This extends from [`kuhn_cfr_notes.md`](kuhn_cfr_notes.md). Read that first for regrets, regret matching, information sets and average strategies.*

CFR+ uses the same counterfactual regret idea as vanilla CFR, but changes three parts of training to improve convergence.

## 1. Clip cumulative regrets at zero

Vanilla CFR keeps negative cumulative regrets:

$$
R_{t+1}(a)=R_t(a)+r_t(a).
$$

CFR+ clips the *stored total* after each update:

$$
R^+_{t+1}(a)=\max\bigl(0,\,R^+_t(a)+r_t(a)\bigr).
$$

For example, starting with regrets `{"check": 4, "bet": 2}` and receiving updates `{"check": -7, "bet": 1}`:

- Vanilla CFR stores `{"check": -3, "bet": 3}`.
- CFR+ stores `{"check": 0, "bet": 3}`.

Key difference: Vanilla CFR already ignores negative regrets *when choosing probabilities*. CFR+ also stops negative cumulative regrets from carrying over to future updates.

## 2. Give later strategies more weight

Vanilla CFR uses equal base weights for training iterations. Our CFR+ implementation uses delayed linear averaging:

$$
w_t=\max(0,t-d),
$$

where $t$ is the iteration and $d$ is the averaging delay. For a delay of 2, the first five weights are 0, 0, 1, 2, 3.

This reduces the influence of early strategies. As in vanilla CFR, our code also accounts for the player's reach probability when averaging.

## 3. Update players one at a time

- Vanilla CFR: Calculate both players' regrets using the same strategy profile, then apply the updates together.
- CFR+: Update Player 0 first, then explore the game again and update Player 1 using Player 0's new strategy.

Our CFR+ implementation therefore makes two full game tree passes per iteration. This can improve convergence but makes each iteration more expensive.

