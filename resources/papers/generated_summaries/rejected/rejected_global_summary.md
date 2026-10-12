# Reliable action recommendations: paper-by-paper summaries

The summaries below provide 2–3 paragraphs per paper. The papers are ordered to reflect thematic dependencies: **context-aware recommendation** (definition, modelling, and scaling), **grounded reasoning and tool use**, and **proactive behaviour and abstention**.

## P. Mateos and A. Bellogín — A systematic literature review of recent advances on context-aware recommender systems (2024/2025)

This systematic literature review surveys recent work on context-aware recommender systems (CARS) and provides a structured vocabulary for describing context. Context is treated as more than a set of features, and is characterised by acquisition (explicit, inferred, sensed), observability, and dynamism. The review additionally groups context into semantic categories such as individual, spatial, temporal, activity, and relational context.

The review highlights an algorithmic shift toward neural and attention-based approaches, and reports that modelling context directly is typically more expressive than pre- or post-filtering strategies. It also identifies methodological limitations in the literature, including fragmented datasets, limited code availability, and a strong reliance on offline ranking metrics. This framing motivates work that embeds context into the decision process explicitly, as in the context-aware MDP formulation used by Bukhari et al.

## M. Bukhari, M. Maqsood, and F. Adil — An actor-critic based recommender system with context-aware user modelling (2025)

This work formulates recommendation as a context-aware Markov Decision Process in which the state includes both recent user interactions and contextual variables (e.g., time, location, or other situational factors). The main contribution is not only the use of actor–critic reinforcement learning, but the explicit construction of context-conditioned states through two strategies: a hard gating mechanism (context-specific zero weighting) and a soft attention mechanism (context-specific attention weighting).

The paper’s broader takeaway is that treating recommendation as sequential decision-making provides a natural interface to action selection in adaptive systems, where the system must choose among alternative next steps under uncertainty. At the same time, it exposes a practical constraint: in many applications, the action set is large. This motivates scalable action-selection mechanisms such as those studied by Dulac-Arnold et al. for large discrete action spaces.

## G. Dulac-Arnold et al. — Deep reinforcement learning in large discrete action spaces (2016)

This paper addresses reinforcement learning settings with extremely large discrete action spaces, which are common in recommender systems and other decision problems. The proposed Wolpertinger architecture combines continuous-action learning with discrete action selection by first producing a continuous proto-action, retrieving a small candidate set using approximate nearest neighbours in an action embedding space, and then selecting among candidates using a critic.

For action recommendation problems, the central insight is that representing actions in an embedding space enables both generalisation and efficient retrieval, avoiding exhaustive scoring over all actions. Although the paper predates LLM-based agents, the same scaling concern arises when an assistant system must choose among many possible tool calls, database operations, and interaction strategies. The next papers (ReAct and Toolformer) focus on how such actions can be grounded in external evidence rather than internal inference alone.
## Y. Geifman and R. El-Yaniv — Selective classification for deep neural networks (2017)

Selective classification formalises the idea that a model should be allowed to abstain rather than predict on every input. The framework separates coverage (the fraction of cases on which the model acts) from risk (the error on the cases it does handle), and provides procedures for selecting confidence thresholds that yield risk guarantees with high probability.

Although presented for deep classifiers, the concept transfers directly to action recommendation in adaptive assistants. In proactive settings, abstention can mean remaining idle, asking for clarification, or acquiring additional evidence via tool calls. Positioned after the proactivity papers, selective classification provides a principled interpretation of restraint: reliable systems should optimise not only for correctness when acting, but also for acting only when sufficient confidence and user acceptance are expected.
