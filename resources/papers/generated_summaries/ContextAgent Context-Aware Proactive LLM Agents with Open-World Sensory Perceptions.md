## Info
- **Title:** ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions  
- **Authors & Year:** Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan — 2025  
- **Venue:** NeurIPS 2025

# 2. Problem Addressed
Most LLM agents today remain reactive. They wait for explicit user instructions and ignore the rich cues from users’ physical environments. This limits autonomy, usefulness, and the ability to assist when users are busy or hands‑occupied. Prior proactive agents typically rely on desktop screenshots or rule-based triggers, leaving a gap in open‑world sensory understanding and tool‑augmented action. The paper tackles the challenge of building an agent that can *understand real‑world context* from wearable sensors and proactively decide when and how to help.

# 3. Key Contributions
- Proposes **ContextAgent**, the first proactive agent that fuses multi‑modal sensory inputs (video, audio, notifications) with persona information.  
- Introduces a **proactive-oriented context extraction** pipeline for high‑quality sensory interpretation.  
- Designs a **context-aware reasoner** with distilled reasoning traces that predicts proactivity needs and plans tool chains.  
- Releases **ContextAgentBench**, the first benchmark for evaluating context-aware proactive agents across 1,000 scenarios and 20 tools.  
- Demonstrates state‑of‑the‑art results, outperforming strong baselines in proactive prediction and tool calling.

# 4. Method / Approach
ContextAgent first processes egocentric video, audio, and smartphone notifications using vision-language models and speech recognition to produce concise, actionable sensory context. It also integrates persona context derived from historical data to capture user preferences and habits. These contexts feed into a fine‑tuned LLM “reasoner” trained with chain‑of‑thought supervision distilled from stronger models. The reasoner outputs a proactive score, explains its internal reasoning, and proposes a sequence of tool calls when needed. If the score exceeds a user‑tunable threshold, the system executes the tool chain and combines tool results with contextual reasoning to form the final response. The entire loop enables end‑to‑end proactive assistance driven by real‑world sensory cues.

# 5. Results & Evidence
Experiments on ContextAgentBench show consistent improvements: up to **+8.5% accuracy in proactive prediction**, **+7.0% F1 on tool calling**, and **+6.0% tool‑argument accuracy** over baselines. The agent maintains strong performance even when compared to 70B‑scale or proprietary LLMs, despite using smaller models. Qualitative examples illustrate correct proactive actions (e.g., predicting bus arrival needs at a stop) and correct restraint in non‑action scenarios. Results on a sensor‑based subset and out‑of‑domain splits confirm robustness and generalization.

# 6. Strengths & Limitations
**Strengths:**  
- Integrates real-world multimodal sensing rather than relying on desktop-only inputs.  
- Thought-trace distillation significantly improves proactive reasoning.  
- Strong benchmark and dataset contributions.  
- Outperforms larger models despite using modest-sized base LLMs.

**Limitations:**  
- Tool set is fixed and relatively small; real-world deployment would require broader API support.  
- Personas depend on high-quality historical data, which may be sparse in practice.  
- Context extraction still relies on VLM quality and may miss subtle cues.  
- Scenarios in the benchmark, though diverse, don’t fully cover long-term, continuous real‑world use.

# 7. Essential Takeaways
- Proactive agents need both **sensor-derived context** and **persona context** to make good decisions.  
- High-quality **context extraction** is just as important as downstream reasoning.  
- **Think‑before‑act** training via reasoning-trace distillation meaningfully boosts performance.  
- Open‑world perception (not just screen-based input) unlocks entirely new proactive behaviors.  
- Tool‑calling is essential: proactivity is not just about prediction but about acting effectively.  
- Smaller LLMs can be competitive when fine‑tuned on well-designed, context‑rich datasets.  
- Proactivity requires careful balance—avoiding intrusive or irrelevant help is as important as acting.

# 8. One-Sentence Essence
A proactive LLM agent becomes genuinely useful only when it can understand real‑world sensory context and act through tools—ContextAgent is a first step toward that vision.