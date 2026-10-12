## Info
- **Title:** ReAct: Synergizing Reasoning and Acting in Language Models  
- **Authors & Year:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao — 2023  
- **Venue:** ICLR 2023  

# 2. Problem Addressed
Large language models can reason (e.g., chain‑of‑thought) and can act (e.g., interact with tools or environments), but these abilities have mostly been studied separately. This separation limits models: reasoning alone often leads to hallucinations, while acting alone lacks high‑level planning. The paper tackles the gap between these capabilities by exploring whether interleaving reasoning and acting can create more reliable, grounded, and capable agents.

# 3. Key Contributions
- Introduces **ReAct**, a prompting framework that mixes free‑form reasoning traces with concrete actions.  
- Demonstrates synergy between reasoning and acting across QA, fact verification, and interactive decision‑making.  
- Shows ReAct outperforms reasoning‑only and action‑only baselines on multiple benchmarks.  
- Provides analyses, ablations, and initial fine‑tuning results showing ReAct’s scalability and interpretability benefits.

# 4. Method / Approach
ReAct expands the model’s action space to include both *actions* and *thoughts*. Thoughts are free‑form reasoning steps added to the trajectory; actions interact with an external environment (e.g., Wikipedia API, simulated household, or shopping site). The model is prompted with human‑written example trajectories that interleave these steps. During inference, the LLM decides when to think and when to act, forming a closed‑loop cycle of observe → think → act. This design grounds reasoning in retrieved observations, and uses reasoning to guide tool use, search, or navigation. The approach is entirely prompting‑based, though the paper also tests small‑scale fine‑tuning to show additional gains.

# 5. Results & Evidence
Across HotpotQA and FEVER, ReAct reduces hallucinations and improves factual grounding, especially when paired with CoT self‑consistency. On interactive tasks (ALFWorld and WebShop), ReAct achieves large jumps in success rate—up to +34% and +10% over strong baselines—despite using only a handful of in‑context examples. Qualitative findings show clearer reasoning chains, better subgoal tracking, and more reliable search behavior. The overall evidence indicates that combining internal reasoning with external actions produces more robust, interpretable task‑solving trajectories.

# 6. Strengths & Limitations
**Strengths:**
- Produces grounded, interpretable reasoning traces.  
- Substantially boosts performance with minimal examples.  
- Generalizes across very different task types.  

**Limitations:**
- Can loop or get stuck due to imperfect reasoning control.  
- Relies on carefully written demonstrations.  
- Action spaces in experiments are simplified and not fully realistic.  
- Still far from domain‑specific SOTA on hard QA benchmarks.

# 7. Essential Takeaways
- Reasoning and acting reinforce each other: reasoning guides what to retrieve, and retrieval grounds reasoning.  
- External tools or environments help correct hallucinations that arise in pure chain‑of‑thought.  
- Sparse, well‑timed reasoning is often enough — dense thoughts are not always necessary.  
- Simple prompting techniques can outperform large imitation‑learning baselines in interactive tasks.  
- Interpretability improves when models externalize their thought process step‑by‑step.  
- Fine‑tuning on generated trajectories can unlock additional gains.  
- Integrating LLMs with actions is a promising direction toward more capable, trustworthy agents.

# 8. One-Sentence Essence
ReAct shows that interleaving natural‑language reasoning with environment‑grounded actions makes language models more accurate, reliable, and useful across both reasoning and decision‑making tasks.