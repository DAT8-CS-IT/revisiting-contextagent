# Reliable action recommendations: paper-by-paper summaries

The summaries below provide 2–3 paragraphs per paper. The papers are ordered to reflect thematic dependencies: **context-aware recommendation** (definition, modelling, and scaling), **grounded reasoning and tool use**, and **proactive behaviour and abstention**.

## S. Yao et al. — ReAct: Synergising reasoning and acting in language models (2023)

ReAct proposes an interaction pattern in which a language model interleaves natural-language reasoning steps with explicit actions that query an external environment (e.g., retrieval or tool use). The approach aims to reduce failures associated with purely internal reasoning, particularly hallucination, by grounding subsequent reasoning in observations obtained through actions.

In the context of reliable action recommendations, ReAct provides an operational interpretation of reliability: a system should be designed to verify and refine its decisions by acting to acquire evidence. This perspective connects naturally to Toolformer, which considers how tool use can be learned as a capability rather than treated solely as a prompting convention.

## T. Schick et al. — Toolformer: Language models can teach themselves to use tools (2023)

Toolformer proposes a self-supervised method for training language models to call external tools. The method generates candidate tool calls, executes them, and retains only those calls whose outputs reduce next-token prediction loss. Fine-tuning on the resulting augmented data teaches the model when to call tools and how to format calls, without requiring large hand-annotated tool-use datasets.

Relative to ReAct, Toolformer shifts the emphasis from prompting trajectories to learning tool invocation policies from data. The paper also supports a practical design principle for adaptive assistants: tool calls should be invoked selectively, only when they improve decision quality. This selective invocation becomes particularly important for proactive systems that operate under sensing and compute constraints, as demonstrated by ContextAgent and ProAgent.

## B. Yang et al. — ContextAgent: Context-aware proactive LLM agents with open-world sensory perceptions (2025)

ContextAgent targets proactive assistance in open-world settings by incorporating multimodal sensory inputs (e.g., egocentric video, audio, notifications) and persona information. The approach emphasises a proactive-oriented context extraction pipeline that converts raw sensory streams into structured, actionable context. A context-aware reasoner then predicts whether proactive intervention is needed and produces tool-oriented plans.

This work connects to grounded acting (ReAct) by treating tool calls as the mechanism by which proactive intent is operationalised, and it contributes an evaluation benchmark focused on proactive prediction and tool-calling accuracy. It also motivates systems-level questions that become central in ProAgent: sensory perception is informative but expensive, and practical deployments require resource-aware sensing and temporal mechanisms to avoid redundant or intrusive behaviour.

## B. Yang et al. — ProAgent: Harnessing on-demand sensory contexts for proactive LLM agent systems (2025)

ProAgent extends the context-aware proactive agent design by focusing on efficiency and deployability. The system combines always-on low-cost sensing with on-demand activation of high-cost perception (e.g., vision), and includes mechanisms such as persona filtering and temporal filtering to reduce repeated or unnecessary interventions. This positions proactivity as a decision problem with explicit computational and interaction costs.

The paper’s takeaways align with the broader goal of reliable action recommendation: proactive assistance requires both accurate context extraction and principled restraint, since false positives can degrade user experience. This connects directly to Lu et al., which frames proactivity in terms of user acceptance, and to Diebel et al., which analyses the human factors underlying satisfaction and perceived competence.

## Y. Lu et al. — Proactive Agent: Shifting LLM agents from reactive responses to active assistance (2024)

This paper studies proactivity as a learnable behaviour using event-driven scenarios spanning multiple domains. Proactivity is framed as predicting which task or assistance action to offer given an evolving environment state, and a reward model trained on human acceptance judgements is used to shape agent behaviour. This aligns evaluation with a user-facing objective: whether an intervention is appropriate and timely.

The key implication is that proactive performance is constrained by false alarms and timing errors as much as by task coverage. This complements ProAgent’s systems view of cost and intrusiveness, and prepares the ground for Diebel et al., which provides empirical evidence that proactive assistance can reduce satisfaction by undermining competence-based self-esteem.

## C. Diebel et al. — When AI-based agents are proactive: implications for competence and system satisfaction in human–AI collaboration (2025)

This study evaluates the effects of proactive (vs. reactive) assistance on user experience in a human–AI collaboration setting. The results indicate that proactive help can reduce system satisfaction, with the effect mediated by reductions in competence-based self-esteem. The negative effect is amplified for users with higher AI knowledge, suggesting that user characteristics shape the acceptability of proactive interventions.

In relation to Lu et al., the findings motivate acceptance-aware designs and configurable proactivity levels rather than universal proactive defaults. The results also motivate formal mechanisms for restraint and abstention: reducing unnecessary interventions can be a reliability feature, not merely a conservative choice. This motivates the selective prediction framework described by Geifman and El-Yaniv.

