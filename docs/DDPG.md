# Deep Deterministic Policy Gradient — a learning guide

This document explains the reinforcement learning background needed to
understand DDPG, and then walks through the specific bugs that stopped the
from-scratch implementation in this repo from converging, and how each one
was fixed. It's meant to be read by someone who knows the basics of neural
networks but is new to RL.

## 1. The reinforcement learning problem

An agent interacts with an environment over discrete time steps. At each step
$t$ it observes a state $s_t$, picks an action $a_t$, and receives a reward
$r_t$ plus the next state $s_{t+1}$. The agent's behavior is a policy
$\pi(a|s)$, and the goal is to find the policy that maximizes the expected
discounted return:

$$
J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right], \qquad \gamma \in [0, 1)
$$

$\gamma$ (the discount factor) trades off immediate vs. future reward, and
also keeps the sum finite for infinite-horizon tasks.

Two classic ways to represent what "good" means:

- **Value-based**: learn $Q(s, a)$, the expected return of taking action $a$
  in state $s$ and acting optimally afterwards. The policy is implicit:
  "act greedily w.r.t. $Q$". This is what DQN does — but taking an argmax
  over a *continuous* action space at every step is not tractable, which is
  exactly the problem DDPG was built to solve.
- **Policy-based**: parameterize the policy directly, $\pi_\theta(a|s)$, and
  climb the gradient of $J$ with respect to $\theta$.

## 2. Why the pendulum needs continuous actions

`Pendulum-v1` applies a torque in $[-2, 2]$ N·m to a free-swinging pendulum;
the goal is to swing it up and hold it upright. There is no natural discrete
set of torques — quantizing the action space throws away the precision
needed for the final balancing act. This is the regime DDPG targets: a
continuous action space, and a policy that outputs one specific action
instead of a probability distribution to sample from.

## 3. Deterministic Policy Gradient

DDPG ([Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)) combines
two ideas:

1. **Deterministic Policy Gradient theorem** (Silver et al., 2014): instead
   of a stochastic policy $\pi(a|s)$, learn a deterministic one
   $\mu_\theta(s)$ that outputs a single action. Its gradient turns out to be:

   $$
   \nabla_\theta J \approx \mathbb{E}_{s}\left[\nabla_a Q(s, a)\big|_{a=\mu_\theta(s)} \; \nabla_\theta \mu_\theta(s)\right]
   $$

   In plain terms: push the actor's output in the direction that increases
   the critic's Q-value the fastest. The critic acts as a learned, local
   approximation of "which way should I move this action to do better."

2. **DQN's stabilization tricks**, adapted to actor-critic: a replay buffer
   and target networks (see below).

This makes DDPG an **actor-critic** method: the actor $\mu_\theta(s)$ picks
actions, the critic $Q_\phi(s, a)$ evaluates them, and each is trained off
the other.

## 4. The four pieces that make it stable

### Replay buffer

Consecutive transitions from a rollout are highly correlated (state at $t+1$
is derived from the state at $t$), which violates the i.i.d. assumption most
gradient-based optimizers rely on. The fix is to store transitions
$(s, a, r, s', \text{done})$ in a large buffer and train on random minibatches
sampled from it — see [`ReplayBuffer`](../src/ddpg/replay_buffer.py).

### Target networks + soft updates

The critic's training target is itself computed by the critic:

$$
y = r + \gamma \, Q_\phi(s', \mu_\theta(s'))
$$

Optimizing $Q_\phi$ towards a target that moves every time $Q_\phi$ moves is
prone to oscillation and divergence — the same issue DQN addresses with a
frozen target network. DDPG instead keeps a **slowly-tracking copy** of both
networks, updated every step by a small fraction $\tau \ll 1$:

$$
\theta' \leftarrow \tau \theta + (1-\tau)\theta'
$$

rather than periodically hard-copying the weights. This keeps the
bootstrap target nearly stationary from one update to the next. See
`DDPGAgent._soft_update` in [`agent.py`](../src/ddpg/agent.py).

### Exploration noise

A deterministic policy never explores on its own — with no randomness, it
keeps producing the same action for a given state forever, including a bad
one it hasn't yet learned to avoid. DDPG adds noise to the action at
collection time only (the underlying policy stays deterministic):

$$
a_t = \mu_\theta(s_t) + \mathcal{N}_t
$$

The paper uses an **Ornstein-Uhlenbeck (OU) process** rather than i.i.d.
Gaussian noise, specifically because OU noise is *temporally correlated*: a
random push at step $t$ tends to keep pushing in the same direction for the
next few steps, instead of averaging itself out. For a torque-controlled
system like a pendulum, that matters — a single independent random torque
sample is rarely enough to build up the momentum needed to swing past
horizontal, but a short correlated burst is. See [`OUNoise`](../src/ddpg/noise.py).

### A warm-up of purely random actions

Before any learning happens, the actor is close to a random function with a
small output (see the initialization note below), so exploration noise added
on top of it only ever probes a narrow band of weak actions. DDPG's original
implementation sidesteps this by taking **actions sampled uniformly from the
full action space** for the first few thousand steps, regardless of what the
actor says. This is what actually gets the pendulum to experience a full
swing at least once, which is a precondition for the critic ever learning
that it's worth doing. This is `initial_random_steps` in
[`DDPGAgent.select_action`](../src/ddpg/agent.py).

## 5. What was wrong with the first from-scratch attempt

The project's initial hand-written notebook implemented the actor, critic,
replay buffer, and even an `OUNoise` class — but the final training loop
plateaued around a reward of **-500** and never produced a visible
swing-up, while the
[reference tutorial notebook](https://github.com/Curt-Park/rainbow-is-all-you-need)
this project also collected reached a properly balanced pendulum. Comparing
the two line by line surfaced four concrete, fixable gaps — none of them
about the core DDPG math, all about the details that make it actually learn:

| # | Issue in the hand-written version | Why it matters | Fix |
|---|---|---|---|
| 1 | No `initial_random_steps` warm-up in the final training cell — actions were `actor(state) + N(0, 0.1)` from step 0 | With a near-zero, untrained actor, small Gaussian noise never explores strong torques, so the agent never observes a full swing-up in its replay buffer to learn from | Added an explicit random-action phase before any policy-driven action selection (§4) |
| 2 | `OUNoise` was implemented but never called — exploration used plain `np.random.normal(0, 0.1)` instead | Uncorrelated noise self-cancels step to step instead of accumulating into a sustained push | Wired the OU process into `select_action` |
| 3 | The `OUNoise.sample()` formula used `random.random()` (uniform on $[0, 1)$, **always positive**) for the Wiener increment, instead of a zero-mean term | Silently biases every noise sample positive, drifting exploration in one direction instead of centering it on the policy's own action | Replaced with `np.random.randn(...)` (zero-mean Gaussian), which is the correct discretization of $dW_t$ |
| 4 | Actor/critic output layers used PyTorch's default initialization | Larger initial weights can push the actor's pre-`tanh` output into the saturated region immediately, flattening gradients before learning starts | Small uniform init (`±3e-3`) on the last layer of both networks, as specified in the original DDPG paper |

None of these are exotic — they're the kind of thing that's easy to skip
when translating an algorithm description into code for the first time,
because the pseudocode versions of DDPG rarely spell out *why* each piece is
there. The training curve in [`assets/training_curve.png`](../assets/training_curve.png)
and the demo video are produced by [`src/ddpg`](../src/ddpg), which includes
all four fixes.

## 6. References

- Lillicrap et al., 2015. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (the DDPG paper).
- Silver et al., 2014. [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf).
- Mnih et al., 2015. [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (DQN, the replay buffer / target network lineage).
- [Curt-Park/rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need) — the reference tutorial notebook used to cross-check this implementation.
- [Gymnasium `Pendulum-v1` source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py) — reward function and dynamics.
