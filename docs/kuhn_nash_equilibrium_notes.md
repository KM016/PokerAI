# Nash Equilibrium

## What is Nash equilibrium?

A Nash equilibrium is when neither player can improve their expected profit by changing their strategy on their own, assuming the other player keeps playing the same way.

This doesn't mean neither player can change their strategy. It means neither has an incentive to, because changing alone wouldn't improve their expected payoff.

In poker, this means mixing actions. For example, bluffing with a Jack 1/3 of the time instead of always bluffing or never bluffing. The randomisation makes behaviour harder to exploit.

The goal is to find action probabilities that work together as a Nash equilibrium, then use them as a benchmark for CFR and RL.


## The main idea behind the probabilities

We choose certain bluffing and calling frequencies so that the other player becomes indifferent between two actions.

*Indifferent* just means both choices have the same expected value (EV). If calling and folding have the same EV, for example, the player has no advantage from changing how frequently they call.

The important bit is that the probabilities must work together across the whole game, not just for one decision.

We will construct one particular Nash equilibrium. We choose the version where Player 0 always bets with a King. 

## 1. How often should Player 0 bluff with a Jack?

Suppose Player 1 has a Queen and Player 0 bets. Player 0 must have either a Jack (bluff) or a King (stronger hand).

First calculate how likely Player 0 is to have a Jack given that they bet.

Let:

- $x=P(\text{Bet}\mid J)$: how often Player 0 bluffs with a Jack.
- $P(\text{Bet}\mid K)=1$: we're assuming Player 0 always bets with a King.

Before seeing the bet, the Jack and King are equally likely ($1/2$ each). Using Bayes' theorem:

$$
P(J\mid\text{Bet})
=\frac{P(\text{Bet}\mid J)P(J)}{P(\text{Bet}\mid J)P(J)+P(\text{Bet}\mid K)P(K)}
$$

$$
=\frac{x(1/2)}{x(1/2)+1(1/2)}
=\frac{x}{x+1}.
$$

Now find the probability of bluffing that makes Player 1 indifferent between calling and folding with a Queen.

Let $p=P(J\mid\text{Bet})$. Calling wins $+2$ against a Jack and loses $-2$ against a King:

$$
EV(\text{Call})=2p-2(1-p)=4p-2.
$$

Folding always loses the original chip:

$$
EV(\text{Fold})=-1.
$$

Make the EVs equal:

$$
4p-2=-1
\quad\Rightarrow\quad
p=\frac14.
$$

So, when Player 1 sees a bet, they must believe there's a 25% chance of a Jack to be indifferent between calling and folding. Substitute this into the Bayes result:

$$
\frac{x}{x+1}=\frac14
\quad\Rightarrow\quad
4x=x+1
\quad\Rightarrow\quad
\boxed{x=\frac13}.
$$

Result: Player 0 bets with a Jack 1/3 of the time and checks the other 2/3. Under our chosen equilibrium, Player 0 bets with a King 100% of the time.

Note : 1/3 is how often Player 0 bluffs when holding a Jack while 1/4 is the probability Player 1 assigns to a Jack after observing a bet.

## 2. How often should Player 1 call with a Queen?

Now Player 0 has a Jack and must choose whether to bluff or check.

Let $q=P(\text{Call}\mid Q,\text{ facing opening bet})$ for Player 1.

If Player 0 bluffs:

| Player 1's card and response | Probability | Player 0 payoff |
|---|---:|---:|
| Queen folds | $\frac12(1-q)$ | $+1$ |
| Queen calls | $\frac12q$ | $-2$ |
| King calls | $\frac12$ | $-2$ |

So:

$$
\begin{aligned}
EV(\text{Bet})
&=\frac12(1-q)(1)+\frac12q(-2)+\frac12(-2)\\
&=\frac12-\frac12q-q-1\\
&=-\frac12-\frac32q.
\end{aligned}
$$

If Player 0 checks with a Jack, they'll lose at showdown against a Queen, and they'll fold if the King bets. Either way:

$$
EV(\text{Check})=-1.
$$

To keep Player 0 indifferent between bluffing and checking:

$$
-\frac12-\frac32q=-1
\quad\Rightarrow\quad
\frac32q=\frac12
\quad\Rightarrow\quad
\boxed{q=\frac13}.
$$

Result: When Player 1 holds a Queen and faces Player 0's opening bet, they call 1/3 of the time and fold 2/3.

## 3. How often should Player 0 call with a Queen after checking?

Now we're looking at a different betting sequence: Player 0 checks, then Player 1 bets.

Suppose Player 1 has a Jack. Under our chosen equilibrium, Player 0 always bets with a King, so if Player 0 checked and Player 1 has a Jack, Player 0 must have a Queen.

Let $q=P(\text{Call}\mid Q,\text{ after check--bet})$ for Player 0.

For Player 1, bluffing with a Jack gives:

$$
\begin{aligned}
EV(\text{Bet})
&=(1-q)(+1)+q(-2)\\
&=1-3q.
\end{aligned}
$$

Checking with a Jack loses to the Queen:

$$
EV(\text{Check})=-1.
$$

Set them equal:

$$
1-3q=-1
\quad\Rightarrow\quad
3q=2
\quad\Rightarrow\quad
\boxed{q=\frac23}.
$$

Result: Player 0 calls with a Queen 2/3 of the time after checking and then facing a bet; otherwise they fold.

This is different from Player 1's 1/3 calling frequency, because the positions and betting histories are different.

## 4. How often should Player 1 bluff with a Jack after a check?

Let $b=P(\text{Bet}\mid J,\text{ Player 0 checked})$ for Player 1. Player 1 always bets with a King in our strategy.

If Player 0 has a Queen, the remaining cards are Jack and King, initially equally likely. After Player 1 bets, Bayes' theorem gives:

$$
P(J\mid\text{Bet after Check})
=\frac{b(1/2)}{b(1/2)+1(1/2)}
=\frac{b}{b+1}.
$$

As before, a Queen facing a bet is indifferent between calling and folding when the probability of a bluff is 1/4:

$$
\frac{b}{b+1}=\frac14
\quad\Rightarrow\quad
4b=b+1
\quad\Rightarrow\quad
\boxed{b=\frac13}.
$$

Result: When Player 0 checks, Player 1 bluffs with a Jack 1/3 of the time and checks 2/3.

## 5. What about the other decisions?

Some choices don't need randomisation:

- Jack facing a bet: Always fold. Calling loses $-2$ because Jack is the weakest card; folding only loses $-1$.
- King facing a bet: Always call. King always wins at showdown, giving $+2$ instead of losing $-1$ by folding.
- Player 0 starting with a Queen: Always check. Betting isn't worth it against the equilibrium opponent: Jack folds and King calls.
- Player 1 with a King after a check: Always bet. Against a Queen, this gives a chance to win an extra chip when they call.
- Player 1 with a Queen after a check: Check in our chosen equilibrium.
- Player 0 starting with a King: We choose to always bet. Checking would also have the same EV in this particular equilibrium, so this is a valid choice, not the only possible choice.

We can verify the Player 0 Queen decision numerically:

$$
EV(\text{Bet with Q})=\frac12(+1)+\frac12(-2)=-\frac12.
$$

If Player 0 checks, against a Jack they earn $+1$ on average; against a King, they call $2/3$ of the time and fold $1/3$:

$$
\begin{aligned}
EV(\text{Check with Q})
&=\frac12(+1)+\frac12\left[\frac23(-2)+\frac13(-1)\right]\\
&=\frac12-\frac56\\
&=-\frac13.
\end{aligned}
$$

Since $-1/3>-1/2$, checking is better.

For Player 0's King, both betting and checking have the same EV against the equilibrium opponent:

$$
EV(\text{Bet with K})=\frac12(1)+\frac12\left[\frac13(2)+\frac23(1)\right]=\frac76,
$$

$$
EV(\text{Check with K})=\frac12\left[\frac13(2)+\frac23(1)\right]+\frac12(1)=\frac76.
$$

So choosing to always bet is allowed.

## 6. The final strategy table

These are the probabilities we store in `NashAgent`. An information set such as `Q|cb` means *I have a Queen - Player 0 checked, then Player 1 bet.

| Information set | Action 1 | Probability | Action 2 | Probability |
|---|---|---:|---|---:|
| `J|` | Check | $2/3$ | Bet | $1/3$ |
| `Q|` | Check | $1$ | Bet | $0$ |
| `K|` | Check | $0$ | Bet | $1$ |
| `J|c` | Check | $2/3$ | Bet | $1/3$ |
| `Q|c` | Check | $1$ | Bet | $0$ |
| `K|c` | Check | $0$ | Bet | $1$ |
| `J|b` | Call | $0$ | Fold | $1$ |
| `Q|b` | Call | $1/3$ | Fold | $2/3$ |
| `K|b` | Call | $1$ | Fold | $0$ |
| `J|cb` | Call | $0$ | Fold | $1$ |
| `Q|cb` | Call | $2/3$ | Fold | $1/3$ |
| `K|cb` | Call | $1$ | Fold | $0$ |

The table is basically information set $\rightarrow$ probability of each legal action. For example, `J|` tells the bot to bet roughly 1/3 of the times it starts with a Jack. It picks randomly each time, it doesn't keep count to force an exact ratio.

## 7. Does this really give a Nash equilibrium?

We can check the overall expected payoff by considering the six possible ordered deals, each with probability $1/6$, when both players use these strategies.

| Player 0 card | Player 1 card | Expected net payoff to Player 0 |
|---|---|---:|
| J | Q | $-2/3$ |
| J | K | $-4/3$ |
| Q | J | $+1$ |
| Q | K | $-5/3$ |
| K | J | $+1$ |
| K | Q | $+4/3$ |

Therefore:

$$
\begin{aligned}
EV(\text{Player 0})
&=\frac16\left(-\frac23-\frac43+1-\frac53+1+\frac43\right)\\
&=\boxed{-\frac1{18}\approx-0.0556\text{ chips per hand}}.
\end{aligned}
$$

Player 1's expected payoff is $+1/18$, since the game is zero sum.

This EV matches the known Kuhn Poker equilibrium value. The equality checks at the mixed decisions, together with checking the remaining decisions, give us a strategy where neither player can improve by changing strategy alone.

## 8. Important: There's more than one equilibrium

We picked one equilibrium by setting Player 0's King betting probability to 1. More generally, Kuhn Poker has a family of equilibrium strategies, with a parameter:

$$
0\leq\alpha\leq\frac13.
$$

For Player 0:

$$
P(\text{Bet}\mid J,\text{ start})=\alpha,
$$

$$
P(\text{Bet}\mid K,\text{ start})=3\alpha,
$$

$$
P(\text{Call}\mid Q,\text{ after check--bet})=\alpha+\frac13.
$$

All the other equilibrium probabilities in our table stay the same. We chose $\alpha=1/3$, giving King bet $1$ and Queen call $2/3$.

Main takeaway: Our Nash agent follows a mathematically derived strategy table. CFR and RL will eventually try to learn strong action probabilities instead of having us type them in. We'll compare what they learn against this equilibrium benchmark.
