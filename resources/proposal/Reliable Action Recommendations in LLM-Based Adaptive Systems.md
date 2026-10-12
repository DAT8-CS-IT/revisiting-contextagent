# Reliable Action Recommendations in LLM-Based Adaptive Systems

## Introduction

Recommender systems are widely used in adaptive software systems to support decision-making under uncertainty. Traditionally, recommender systems focus on suggesting items such as products, media content, or services by learning user preferences from historical interaction data. In such systems, users, items, and contextual information are typically represented as embeddings in a latent vector space, and recommendations are produced by matching user and item representations using similarity or ranking functions.

Recently, large language models (LLMs) have been increasingly used as central components in interactive and adaptive systems. Beyond generating text, LLMs are now capable of orchestrating tool usage, interacting with external databases, and maintaining conversational context over extended interactions. In these systems, the model repeatedly decides which action to take next, such as calling an external tool, querying or updating a database, asking the user for clarification, or remaining idle.

From a recommender systems perspective, such decision-making can be interpreted as a **context-aware recommendation problem**, where the items to be recommended are **actions** rather than content. The context includes multimodal information such as user interaction history, time, location, and system state. The quality of the system then depends on how reliably the correct action is selected given the current context.

## Problem

The goal of this project is to design and implement an executive assistant-like system that supports both user-directed interaction and autonomous, proactive behaviour. The assistant should operate in a traditional interactive mode, such as chatbot-style communication, user-initiated queries, and simple automations, while also acting without explicit prompts when relevant.

The assistant is intended to interact with multiple system components. These include conversational context, a persistent database for tasks and knowledge, external tools accessed through APIs, such as calendar management or information retrieval, and time-based triggers. As a result, the system has access to a large and heterogeneous set of possible actions at any point in time.

A central challenge in such a system is deciding which action, if any, should be taken given the current context. This is particularly important for proactive behaviour. Excessive or poorly timed user interaction can reduce the usefulness of the assistant and negatively affect the user experience. The system must therefore be able to select appropriate actions while often choosing to remain idle.

In this project, this decision problem is addressed by introducing a recommender system as an explicit action-selection layer within the assistant architecture. The recommender system selects among available actions, such as querying or updating the internal database, calling external tools, interacting with the user, or taking no action. The selection is based on the current context and interaction history. The chosen action is then executed by a large language model, which handles reasoning, tool orchestration, and user communication.

This setting combines several challenges relevant to recommender systems. These include a dynamic action space, an evolving user state, and changing internal representations as the assistant manages its own data over time. The project assesses the suitability of recommender systems for action selection in this context and analyses their role in enabling useful behaviour while maintaining robust and non-intrusive interaction.

## Relevant Work

- B. Yang *et al.*, **“ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems”** (2025).  
  [https://arxiv.org/abs/2512.06721](https://arxiv.org/abs/2512.06721)

- B. Yang *et al.*, **“ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions”** (2025).  
  [https://arxiv.org/abs/2505.14668](https://arxiv.org/abs/2505.14668)

- Y. Lu *et al.*, **“Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance”** (2024).  
  [https://arxiv.org/abs/2410.12361](https://arxiv.org/abs/2410.12361)

- M. Bukhari, M. Maqsood, and F. Adil, **“An actor-critic based recommender system with context-aware user modeling,”** *Artificial Intelligence Review* (2025).  
  [https://doi.org/10.1007/s10462-025-11134-9](https://doi.org/10.1007/s10462-025-11134-9)

- P. Mateos and A. Bellogín, **“A systematic literature review of recent advances on context-aware recommender systems,”** *Artificial Intelligence Review* (2024).  
  [https://link.springer.com/article/10.1007/s10462-024-10939-4](https://link.springer.com/article/10.1007/s10462-024-10939-4)

- C. Diebel, M. Goutier, M. Adam, and A. Benlian, **“When AI-Based agents are proactive: Implications for competence and system satisfaction in Human–AI Collaboration,”** *Business & Information Systems Engineering* (2025).  
  [https://link.springer.com/article/10.1007/s12599-024-00918-y](https://link.springer.com/article/10.1007/s12599-024-00918-y)