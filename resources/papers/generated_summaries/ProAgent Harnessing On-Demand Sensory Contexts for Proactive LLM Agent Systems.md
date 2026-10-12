## Info
- **Title:** ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems  
- **Authors & Year:** Bufang Yang, Lilin Xu, Liekang Zeng, Yunqi Guo, Siyang Jiang, Wenrui Lu, Kaiwei Liu, Hancheng Xiang, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan — **2025**  
- **Venue:** arXiv (Preprint)

# 2. Problem Addressed
Most LLM agents today operate reactively: they wait for explicit user instructions, which adds physical and cognitive overhead and makes them miss timely opportunities to help. The authors argue that, in real-world mobile or wearable settings, agents should proactively sense and interpret massive multimodal contexts—vision, audio, motion, location, and personas—to anticipate user needs. Prior work either relied on narrow environments (desktop UIs) or lacked robust reasoning from noisy sensor data. This paper fills that gap by designing an end-to-end agent that continuously perceives the world and provides proactive, unobtrusive help.

# 3. Key Contributions
- **ProAgent**, the first full-stack proactive LLM agent integrating rich multisensory perception with LLM/VLM reasoning.  
- A **proactive-oriented context extraction pipeline** that structures multimodal signals into hierarchical sensory + persona contexts.  
- **Context-aware proactive reasoner** with CoT distillation that predicts user needs and performs tool calling.  
- **On-demand tiered perception** combining always-on low-cost sensors with adaptive high-cost vision sampling.  
- Comprehensive evaluation on real-world AR glasses + edge devices, public datasets, and a user study.

# 4. Method / Approach
ProAgent runs a multimodal perception stack that continuously collects low-cost sensor data (location, motion, audio) and selectively triggers high-cost video sensing based on contextual cues. The system converts raw signals into structured sensory contexts—textual descriptions of location, movement state, dialogue snippets, and visual summaries. It retrieves only scenario-relevant personas using a coarse-grained scene classifier to avoid overhead. A VLM-based proactive reasoner then jointly processes images and contexts, generates intermediate “thoughts,” predicts the urgency of proactive intervention, and—if needed—calls external tools (weather, calendar, transport, etc.). CoT distillation helps the model learn to verbalize situational awareness before making decisions. Temporal constraints avoid repeated or redundant assistance during stable situations.

# 5. Results & Evidence
Experiments on a public dataset and a real-world AR-glasses testbed show that ProAgent boosts proactive-need prediction accuracy by **up to 33.4%**, and tool-calling F1 by **16.8%**, compared to strong baselines. It reduces memory usage and token consumption by unifying vision interpretation and reasoning into one VLM. User studies report higher satisfaction, timeliness, and relevance of assistance. Qualitative cases show improved recognition of needs (e.g., noticing a missed bus or contextual shopping help) and fewer false alarms.

# 6. Strengths & Limitations
**Strengths:**  
- High real-world practicality with AR glasses + edge deployment.  
- Strong integration of multimodal sensing with efficient reasoning.  
- Significant improvements in prediction accuracy and tool use.  
- Careful user study validating usefulness and reduced intrusiveness.

**Limitations:**  
- Still prone to occasional false positives/negatives in ambiguous contexts.  
- Performance depends heavily on VLM size and quality.  
- Persona descriptions must be accurate; incomplete profiles weaken predictions.  
- Limited by toolset coverage—cannot help when user needs exceed available APIs.

# 7. Essential Takeaways
- Proactivity requires agents to *perceive first*, not just respond to text.  
- Multimodal signals provide rich but noisy evidence; structuring them is key.  
- Persona-aware filtering improves accuracy while keeping prompts small.  
- Unified VLM pipelines reduce overhead versus two-stage LLM + VLM systems.  
- On-demand sensing dramatically cuts resource cost without losing recall.  
- Temporal filtering prevents over-assistance and improves user comfort.  
- Real-world deployment needs both algorithmic finesse and systems optimization.

# 8. One-Sentence Essence
ProAgent shows that combining multimodal context sensing with efficient VLM reasoning enables LLM agents to provide timely, proactive, and low-overhead assistance in everyday life.