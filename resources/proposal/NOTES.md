# Problem formalization

## 0. LLMs + Basic Building Blocks as self-assembling Software Systems (not suitable as research question rn)
Can an LLM-based system equipped with a set of basic building blocks (function calling, state manager, temporal execution, self-managed storage, etc) mimic the capabilities of explicitly developed software systems and their business logic.

## 1. Recommender Systems in LLM Action Spaces

1. Function calling (Calling API's, running tools, accessing databases)
2. System-initiated interactions (Notifications, reminders, proactive suggestions)

At each step, the system must decide whether to act and which action to choose from a set of available actions that may change over time.
So the problem is essentially: Given partial and changing context, the system must choose an action or remain idle in order to provide value(define what value is) to the user, while avoiding being annoying af.

## Common Challenges
Some challenges occur in this setting:

1. Dynamic action space (Set of available actions are not fixed. Tools and capabilities may be added, removed or modified)
2. Evolving user state (User goals, preferences and tolerance for interruptions change over time and interactions)
3. Changing internal representations (Representations used by the LLM, like embeddings or tool descriptions may change when model or propmts are modified)
4. Deciding when to remain idle and avoiding over-engagement (Acting too often is annoying = reduce trust. System must learn when to do nothing)

## Techniques and possible features
To address these challenges, the system relies on the following ideas/features:

### Decision mechanisms
1. Context-aware ranking 
2. Confidence thresholding

### Modeling components
3. Temporal modeling
4. Multi-modal context fusion: integrate conversational, database, and system state information

## Data inputs

- Conversational context: User message history and dialogue state
- User environment context: Current User activity, location, DND?, device screenshots
- Temporal information: Time-of-day, date, historical events
- User interaction history: Past actions, queries, past engagement patterns
- Database state: Task records, knowledge base, user profile data
- System state: Available tools, location, resource constraints
- External signals: Calendar events, notifications, API responses

Note: Not all inputs are assumed to be available at all times. The system is expected to work with incomplete context.

## Implementation questions and details
- Integrating with existing tools vs bespoke everything?
    - Meaning, limited customisation vs burden of implementation
- Use custom representation and logic
    - Can offer more control, but increase complexity

## Supervisor red flags

### This is about agents, recommender systems, user modeling, multimodal context ect. is this realistically doable?
This project focuses on a limited set of tools and interaction types in a controlled experimental setting, rather than a full production-scale assistant.
(we just need to find them) IMPORTANTANTE!

### Where is the actual research contribution?
Like how do we measure doing nothing is good? and how do we evaluate it?
Measure the effectiveness of abstention evaluated in terms of reduced unneccesarry actions, without loss of success.

### User environment = privacy concerns ?
Things like location, device state, screenshots. = feasibility and ethics problem

Explicitly say that all context signals are assumed synthetic, anomyized or simulated where necessary.

### Do these datasets actually support proactive action selection?
Our datasets are valid, but mostly function-calling focused, so not really proactive.. but:

We could acknowledge it and add synthetic scenarios to be able to study it.

We do have ContextAgentBench, a dataset + benchmark tool for model proactiveness, rather than function calling. Proactiveness here does not explicitly mean to interact with the user, but it's a start. The dataset looks like this:
| Field | Type | Description |
|-------|------|-------------|
| **Category** | string | Scenario type: `Work`, `Shopping`, `Chitchat`, etc. |
| **Vision** | string | Description of visual context from the environment |
| **Audio** | string | User's spoken input/dialogue |
| **Context information** | string | Combined synthesis of vision + audio context |
| **Mobile api data** | array | Available mobile/API data (usually empty) |
| **Personas** | array | User role/context descriptions |
| **Thoughts** | string | Agent's reasoning about what tools are needed |
| **Proactive index** | boolean | `true`/`false` - should the agent be proactive? |
| **Proactive score** | integer | 1-5 scale: degree of proactivity required |
| **Tool planning** | string | Comma-separated tool names e.g., `<get_city_weather>,<check_agenda_time_conflict>` |
| **Action** | array/string | Specific steps for tool usage |
| **Response** | string | Agent's natural language response to user |
| **Tools** | string (JSON) | Array of tool objects with `name`, `description`, `parameters` |

## Actionable - Proactive LLMs
- Get ContextAgentBench working
- Benchmark combinations of LLMs and techniques
- Techniques:
    - Zero/one/multi-shot learning (In-Context Learning - paper)
    - Inference-time Semantic steering
        - [How-to](https://www.youtube.com/watch?v=F2jd5WuT-zg)
        - [Theory](https://huggingface.co/spaces/dlouapre/eiffel-tower-llama)
    - Low Rank adaptation
    - Fine-tuning (Supervised Fine-tuning - paper)
- MCP instead of direct tool-calling (Future work - paper)
- Enhance dataset (Only has 9 life scenarios - paper)
## Resources

### Datasets

- [When2Call - Nvidia](https://github.com/NVIDIA/When2Call)
- [Toucan 1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)
- [Synthesizing Function Calling Datasets](https://www.emergentmind.com/topics/synthetic-function-calling-dataset)
- [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://apigen-pipeline.github.io/)

#### Synthetic generation
- [Distilearn](https://github.com/argilla-io/distilabel)
- [DataDreamer](https://github.com/datadreamer-dev/DataDreamer)
- [Open AgentInstruct](https://github.com/ThomasRochefortB/open-agentinstruct)

### General
- [Function Calling Directory](https://github.com/Applied-Machine-Learning-Lab/Awesome-Function-Callings)
- [OpenViking - Context database](https://github.com/volcengine/OpenViking)
- [NLP-based Recommender Systems](https://mobidev.biz/blog/how-to-build-text-based-recommender-system-with-nlp)
- [Embedding-based tool selection](https://zarar.dev/embedding-based-tool-selection-for-ai-agents/)
- [How Copilot selects tools](http://github.blog/ai-and-ml/github-copilot/how-were-making-github-copilot-smarter-with-fewer-tools/)
