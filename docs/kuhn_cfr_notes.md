# Counterfactual Regret Minimisation (CFR)

## What is CFR?

Counterfactual Regret Minimisation (CFR) is an algorithm that learns how often to take each action in a game where players don't know everything, like poker.

With our Nash agent, We calculated the action probabilities and put them into a lookup table. With CFR, we don't give it those probabilities. It starts with a basic strategy, evaluates alternative decisions, and updates its strategy repeatedly.

The aim is for its average strategy to approach a Nash equilibrium: a strategy that an opponent can't profitably exploit. In a two player zero sum game like Kuhn Poker, CFR has theoretical guarantees for this as training increases.

## 1. What does CFR actually learn?

CFR learns a separate probability distribution for each information set (my card + the betting history). For example:

```python
"J|": {"check": 0.5, "bet": 0.5}
```

This means I have a Jack, nobody has acted, and the current strategy checks and bets equally often. At the start of training, we can use 50/50 for every decision because we haven't learned anything yet.

CFR changes those probabilities based on which actions have accumulated positive regret. It doesn't need to know the Nash probabilities in advance.

## 2. What is regret?

Regret measures how much better an action would have been, compared with the expected payoff of the strategy currently being used.

Suppose, in one situation, we've worked out:

| Action | Expected payoff |
|---|---:|
| Check | +0.2 chips |
| Bet | +0.8 chips |

Our current strategy chooses each action 50% of the time, so:

$$
EV(\text{Strategy})=0.5(0.2)+0.5(0.8)=0.5.
$$

The regret for each action is:

$$
r(a)=EV(a)-EV(\text{Strategy}).
$$

So:

$$
\begin{aligned}
r(\text{Check})&=0.2-0.5=-0.3,\\
r(\text{Bet})&=0.8-0.5=+0.3.
\end{aligned}
$$

Interpretation: Betting would have been better than our current mixed strategy, while checking would have been worse. Full CFR also adjusts for how likely the information set is reached.

CFR adds each iteration's regrets to its existing totals. These are called cumulative regrets.

## 3. Regret matching — turning regrets into a strategy

This is the part is implemented in `poker_ai/algorithms/cfr.py`.

Regret matching gives more probability to actions with higher positive cumulative regret:

$$
\sigma(a)=\frac{\max(R(a),0)}{\sum_b\max(R(b),0)},
$$

where $R(a)$ is the cumulative regret for action $a$, and $\sigma(a)$ is its probability in the current strategy.

Example:

```python
regrets = {"check": 4, "bet": 6}
```

$$
P(\text{Check})=\frac{4}{4+6}=0.4,
\qquad
P(\text{Bet})=\frac{6}{4+6}=0.6.
$$

So the current strategy becomes:

```python
{"check": 0.4, "bet": 0.6}
```

Three rules to remember:

- Negative regrets count as zero when calculating action probabilities. For `{"check": -3, "bet": 7}`, the probabilities become 0% Check and 100% Bet.
- If every regret is zero or negative, choose uniformly: 50/50 for two actions.
- In vanilla CFR, negative cumulative regrets are still stored. We only clip them to zero when calculating probabilities - we don't permanently erase them. (CFR+ handles regrets differently.)

## 4. Why is it called *counterfactual*?

*Counterfactual* means "what if I'd taken the other action?"

Suppose I have a Queen and face a bet: information set `Q|b`. I can't see whether the opponent holds Jack or King. CFR therefore explores both possible hidden cards and the possible outcomes of calling and folding.

It uses expected payoffs to ask:

> If I always called here, how much would I expect to earn? What if I always folded instead?

It then compares both answers with the EV of its current mixed strategy.

The extra detail: When updating regret at an information set, CFR weights the payoff differences by how likely the opponent's actions and chance events make that information set possible. This is called the *counterfactual reach probability*. It does not weight that regret by the player's own earlier probability of reaching the information set.

That distinction lets CFR learn a single strategy for `Q|b` across all relevant hidden-card situations, without cheating by observing the opponent's card.

## 5. How one training iteration works

In vanilla, full tree CFR, an iteration roughly does this:

1. Start from the game tree and consider every possible card deal.
2. At every decision, look up the current strategy for that information set.
3. Explore the possible actions and calculate their expected payoffs all the way to the end of each hand.
4. Work backwards: compare each action's EV with the current strategy's EV, and add the appropriately weighted counterfactual regret.
5. Use regret matching to get updated action probabilities for the next iteration.
6. Record the current strategies for the average-strategy calculation.

Then repeat for many iterations. Because Kuhn Poker is tiny, we can examine the full game tree rather than needing to sample random hands for each update.

## 6. Current strategy vs average strategy

This is important: the strategy from the very last iteration is not necessarily the one we evaluate.

- Current strategy: The probabilities produced by regret matching from the latest cumulative regrets.
- Average strategy: A weighted average of the strategies used during training. It is the main output of vanilla CFR.

In standard CFR, the averaging weights account for the probability that the player's own earlier decisions reach the information set. So it isn't always a simple unweighted average of the stored numbers.

We will measure the average strategy's exploitability in `evaluate.py`. As it approaches zero, we're approaching equilibrium play.