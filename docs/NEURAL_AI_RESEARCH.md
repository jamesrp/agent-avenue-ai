# Neural AI research notes

**Date:** August 29, 2026
**Scope:** research and planning only; no learned-agent implementation is part of this document.

## Executive conclusion

The first learned Agent Avenue opponent should be a small **action-conditioned value model**:

```text
safe acting-player observation + one semantic candidate action
    -> Linear(D, 128) -> tanh -> Linear(128, 1) -> sigmoid
    -> probability that the acting player eventually wins
```

This keeps the useful part of Keldon Jones's Race for the Galaxy design—a compact evaluator used to
rank shallow candidate continuations—while making the information boundary explicit for an
imperfect-information game. Train it from terminal outcomes first, using frozen-checkpoint,
offline iterations. Do not begin with AlphaZero-style MCTS, online TD updates, a policy head, or a
belief-state solver.

This is a trusted-code architecture and regression boundary, not a sandbox against hostile in-process
Python. Untrusted agents would need process isolation and an allowlisted message protocol.

The key safety decision is that inference scores `Q(observation, action)`. It must **not** apply each
candidate to the real hidden `GameState` and let the model inspect the realized successor. That
would let recruit choices depend on the actual face-down card and let offer choices depend on the
opponent's actual private hand or future private draws.

## Sources reviewed

The Keldon review used both:

- the official `rftg-0.9.4` source archive linked from <https://www.keldon.net/rftg/>; and
- the public `keldon` branch at commit
  [`902d6c75a5a28171675f0beffb3756cc04dd5a61`](https://github.com/bnordli/rftg/tree/902d6c75a5a28171675f0beffb3756cc04dd5a61).

The repository identifies `ai.c` as Keldon Jones's 2009–2011 code with later modifications by B.
Nordli. Statements below describe that public code, not an attempt to reconstruct every historical
training run.

Primary modern references are listed at the end.

## What the Race for the Galaxy AI actually does

### Two small, dense neural networks

The base two-player checkpoint headers declare:

- evaluator: `704 -> 50 tanh -> 2-way softmax` (`network/rftg.eval.0.2.net`);
- role predictor: `605 -> 50 tanh -> 7-way softmax` (`network/rftg.role.0.2.net`).

The evaluator's committed checkpoint reports 30,000 training iterations. The implementation is a
custom dense network in [`src/net.c`](https://github.com/bnordli/rftg/blob/902d6c75a5a28171675f0beffb3756cc04dd5a61/src/net.c): tanh hidden units, normalized exponential
outputs, hand-written gradient accumulation, and a cache-friendly incremental first layer. Fifty
hidden units are small; the important point is that the input is heavily engineered and candidate
moves are simulated before evaluation.

### Viewpoint-relative, domain-specific inputs

[`setup_nets()`](https://github.com/bnordli/rftg/blob/902d6c75a5a28171675f0beffb3756cc04dd5a61/src/ai.c#L1027-L1390) builds named features for:

- game clock, VP pool, end conditions, and goals;
- exact own-hand card designs and buildable-card counts;
- each player's active cards, goods, hand size, cards seen, military, developments, worlds, and
  consume capacity;
- relative deficits to category leaders; and
- raw one-round role scores used by the role-prediction network.

The evaluated player is placed first as “Us,” followed by opponents, so the representation is
viewpoint-relative rather than tied to a fixed seat. Many scalar quantities use threshold features
such as “at least N cards” rather than one raw number. This gives a tiny hidden layer direct access
to strategically meaningful nonlinearities.

[`eval_game()`](https://github.com/bnordli/rftg/blob/902d6c75a5a28171675f0beffb3756cc04dd5a61/src/ai.c#L2281-L2556) explicitly omits other players'
private hand identities during simulation. Its returned action score is not just the network
probability: it adds tiny hand-size/point tie-break terms and a winner bonus.

### Shallow simulation does much of the work

The code copies the game, installs AI controllers, completes the current phase or round, and then
calls the evaluator. Candidate role choices are integrated over predicted opponent-role
combinations; in a two-player game it checks all opponent-role possibilities. Numerous tactical
choices—discarding, placing, consuming, producing, and others—have dedicated simulation code.

This is the most transferable lesson: the AI is not “a 50-node network that directly picks a move.”
It is a rules-aware candidate generator and simulator with a compact learned leaf evaluator. The
small network works because the engine and hand-written feature/simulation layer present it with a
much easier problem.

### A second network models opponent role choices

The role network predicts a distribution over role choices from public features plus evaluator
scores of raw role continuations. During action selection, the AI:

1. predicts each opponent's role distribution;
2. simulates combinations of opponent roles;
3. completes the round for each own candidate role;
4. weights evaluator scores by the predicted combination probabilities; and
5. takes the highest-scoring role.

The role network is then trained toward a sharp softmax of the expensive simulated action scores.
This is closer to search distillation/opponent modeling than to a modern end-to-end policy head.
Agent Avenue does not need an equivalent second network initially: the response to an offer is an
explicit subsequent decision, and a direct candidate-value model can learn the continuation.

### Online, TD-like self-play training

At the beginning of each round, [`perform_training()`](https://github.com/bnordli/rftg/blob/902d6c75a5a28171675f0beffb3756cc04dd5a61/src/ai.c#L2562-L2620) stores the current evaluator input
and trains older positions belonging to that player toward the current prediction. Training weight
decays backward by `0.7` per stored position. At game end, the target is a softmax over final VP plus
a winner bonus, and the same backward-decayed update is applied.

This is best described as a TD/eligibility-trace-like backward credit assignment scheme, not a
textbook replay-buffer TD(lambda) implementation. The evaluator is updated online while games are
being generated. All AI seats for one game configuration share the same global network, and weights
are saved at shutdown. There is no frozen actor/learner generation boundary.

If no evaluator checkpoint exists, [`initial_training()`](https://github.com/bnordli/rftg/blob/902d6c75a5a28171675f0beffb3756cc04dd5a61/src/ai.c#L8827-L8909) runs 5,000 synthetic terminal
examples with random scores so that winning and being ahead start with sensible value. This is a
useful precedent for warm-starting from known domain structure, though Agent Avenue can first use
its existing heuristic as a data-generating policy instead of fabricating targets.

### Hidden information is handled pragmatically, not generally

The public code takes several practical precautions:

- evaluator features hide simulated opponents' exact hand identities;
- “fake hand” and “fake discard” counts represent uncertain cards;
- selected tactical routines sample unknown cards for explore or opponent-placement decisions; and
- simulations use a fixed synthetic random seed rather than advancing the real one.

It is not a general public-belief or information-set search algorithm. The full authoritative game
is copied into simulations, and uncertainty is handled by bespoke logic where the implementation
needs it. That was a reasonable engineering choice for the project, but it is not a safe abstraction
to copy wholesale into this repository.

### No general exploration policy or promotion gate

Primary decisions are greedy with respect to simulated evaluator scores. The source has randomness
for card sampling and game setup, but no general epsilon-greedy self-play policy, frozen checkpoint
generations, held-out promotion arena, or confidence-based champion gate. Those are areas where a
modern implementation can be simpler to audit and more reproducible.

## Lessons to transfer

1. **Spend representation on rules, not network depth.** Counts, score thresholds, card identities,
   role/phase, and visibility should be explicit.
2. **Canonicalize by viewpoint.** Encode self/opponent, never player one/player two.
3. **Enumerate the small legal set.** Agent Avenue has at most 12 legal offers and two recruit
   choices, so one batched forward pass can score every candidate.
4. **Keep semantic actions outside the network.** Stable `PlayOfferAction` and `RecruitAction`
   values remain the policy interface.
5. **Use a learned scalar to compare candidates.** A fixed policy-output vocabulary is unnecessary
   for the first baseline.
6. **Bootstrap sensibly.** Existing heuristic self-play with explicit exploration is the analogue
   of Keldon's hand-built initial training.
7. **Profile the whole decision path.** Encoding and rules/replay validation may cost more than a
   roughly 11,400-parameter MLP.

## Lessons not to copy

1. Do not use global mutable networks updated during gameplay.
2. Do not let model scores depend on a candidate's actual hidden-state successor.
3. Do not build a second opponent-action network before the scalar baseline shows a need.
4. Do not reproduce hundreds of choice-specific simulation routines for a game with two compact
   semantic decision types.
5. Do not use unversioned checkpoint text files or implicit feature ordering.
6. Do not claim strength without paired, seat-balanced arenas and uncertainty.

## Why the authoritative-successor shortcut leaks

Agent Avenue's engine can deterministically apply a candidate because it knows the deck, both hands,
and the face-down offer. That does not mean the candidate selector may use the resulting state.

### Offer decision

The offerer knows both offered cards, but does not know the opponent's hand or the cards it will draw
after making the offer. If candidate scoring applies the action to the true state and then evaluates:

- the next actor's observation, it exposes that actor's actual private hand to a value computation
  used by the offerer; or
- the offerer's post-action observation, it exposes the actual newly drawn cards.

Either makes the offer depend on hidden information unavailable when the offer was chosen.

### Recruit decision

The recruiter does not know the face-down card. Applying each recruit candidate to the true state
reveals that card through recruited cards, score effects, terminal status, and the next observation.
Choosing the better realized successor is equivalent to looking at the face-down card before
choosing.

A trusted runner may use hidden state to execute the selected action and to produce the eventual
Monte Carlo outcome label. It may not use the realized hidden branch to compare candidates.

### Safe formulation

For acting player `i`, encode only:

```text
Q_theta(o_i, a) = P(i eventually wins | information available in o_i,
                    choose semantic action a,
                    named continuation policy)
```

The hidden face-down card and future draws appear only as environment randomness in the training
outcome. The model learns their policy-conditioned expectation from many games. A later explicit
belief model may marginalize hidden possibilities, but a naïve uniform “unknown card” average is not
exact because the opponent's strategic offer choice carries information about the card.

## Method comparison

### Monte Carlo action-value regression — build first

For every actual decision in a completed game, reconstruct the acting player's safe observation,
combine it with the chosen semantic action, and label it `1` if that player eventually won and `0`
otherwise.

Advantages:

- the target is final and auditable;
- hidden-state aliasing becomes ordinary outcome variance rather than leakage;
- no target network, Bellman max, or online trace implementation is needed;
- the game is short: committed baselines average about 13.3 decisions; and
- 20,000 generated games across the planned generations yield roughly 265,000 retained decision
  examples at committed baseline lengths, with about 53,000 fresh examples per 4,000-game fit.

The main weakness is action coverage: only selected candidates receive outcomes. Seeded exploration
and iterative on-policy data generation are therefore mandatory.

### TD(lambda) or fitted Q targets — first controlled experiment after MC

TD-style targets could reduce terminal-outcome variance and are supported by the TD-Gammon and
Keldon precedents. They also add bootstrapping bias, a moving target, and off-policy maximization.
If added, compute lambda returns over complete stored trajectories using a frozen target checkpoint;
do not add online eligibility traces to the first trainer.

A sensible experiment is a blended loss with 25–50% terminal Monte Carlo target, `gamma = 1`, and
`lambda` in the 0.5–0.8 range. Keep it only if it wins a predeclared arena comparison against the MC
checkpoint on the same corpus and seeds.

### AlphaZero-style policy/value plus MCTS — not yet

AlphaZero combines a policy/value network with repeated MCTS simulations in perfect-information
board games. Agent Avenue's local branching factor is already tiny, so direct candidate scoring is
cheap. More importantly, ordinary state-tree MCTS is invalid here: one public observation
corresponds to many hidden hands, deck orders, and face-down cards.

A valid search upgrade needs information-set or public-belief reasoning, as in work such as ReBeL,
not “determinize the actual hidden state and search it.” It would also spend the CPU budget on many
correlated node visits instead of additional independent self-play games. Search should be revisited
only after the safe value baseline plateaus.

### Search distillation — plausible later upgrade

A shallow, information-safe planner could eventually evaluate one or two future semantic decisions
under sampled or learned beliefs and distill its action rankings into the same candidate network.
This is closer to the useful part of Keldon's role predictor than a full AlphaZero rewrite. It is
out of scope until candidate-value calibration and self-play promotion are reliable.

## Recommended baseline

- Input: versioned flat encoding of the **current acting player's** `PlayerObservation` plus one
  legal semantic candidate action.
- Model: one hidden layer, 128 tanh units, one scalar logit; sigmoid only for probabilities.
- Target: terminal win/loss from that acting player's perspective.
- Policy: batch all legal candidates, greedily select the maximum, and use explicit agent RNG for
  exploration and exact ties.
- Data generation: heuristic warm start, then frozen-checkpoint self-play with seeded exploration.
- Training: offline AdamW/BCE first; no online updates during a game.
- Evaluation: parent, heuristic, and random arenas with paired seeds, alternating seats, confidence
  intervals, and a separate final holdout.
- Artifacts: versioned corpus, dataset, encoder, model, checkpoint, and lineage manifests.

## Primary references

- Keldon Jones, Race for the Galaxy AI project and source: <https://www.keldon.net/rftg/> and
  <https://github.com/bnordli/rftg/tree/902d6c75a5a28171675f0beffb3756cc04dd5a61>
- Gerald Tesauro, “Temporal Difference Learning and TD-Gammon” (1995):
  <https://doi.org/10.1145/203330.203343>
- Richard Sutton, “Learning to Predict by the Methods of Temporal Differences” (1988):
  <https://doi.org/10.1007/BF00115009>
- David Silver et al., “Mastering the Game of Go with Deep Neural Networks and Tree Search” (2016):
  <https://www.nature.com/articles/nature16961>
- David Silver et al., “A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and
  Go through Self-Play” (2017/2018): <https://arxiv.org/abs/1712.01815>
- Johannes Heinrich and David Silver, “Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games” (NFSP, 2016): <https://arxiv.org/abs/1603.01121>
- Noam Brown et al., “Combining Deep Reinforcement Learning and Search for Imperfect-Information
  Games” (ReBeL, 2020): <https://arxiv.org/abs/2007.13544>
- Thomas Anthony, Zheng Tian, and David Barber, “Thinking Fast and Slow with Deep Learning and Tree
  Search” (Expert Iteration, 2017):
  <https://papers.nips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html>
