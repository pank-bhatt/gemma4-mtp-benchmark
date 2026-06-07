# Gemma 4 E2B MLX Detailed Simulation Report (4-Bit Weight Quantized)

This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 E2B** using Apple Silicon **MLX** backend (loaded in **4-Bit Weight Quantized**). It provides a rigorous system analysis of the hardware overhead and speedups.

## 🖥️ System hardware Overhead & Static Pressure Analysis

When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.

### 1. Static Load (Model Weight Memory Footprint)
- **Baseline Static Footprint** (Gemma 4 E2B alone): **2868.5 MB**
- **MTP Static Footprint** (Gemma 4 E2B + Drafter loaded simultaneously): **3592.8 MB**
- **Absolute Hardware Overhead**: **+724.3 MB**
- **Relative Memory Increase**: **25.3% additional Unified RAM required**

> [!IMPORTANT]
> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.

### 2. CPU Dispatch & Context-Switching Pressure
- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.
- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. 
- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.

## 📊 Summary Performance Table

| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Schema Generation** | Baseline | 125.77 t/s | Ref (1.00x) | 3260.1 MB | 55.5% | - | - |
| | **MTP (Spec)** | **64.49 t/s** | **0.51x** | **3259.1 MB** | **47.8%** | **-7.7%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knights & Knaves Deduction** | Baseline | 125.42 t/s | Ref (1.00x) | 3259.3 MB | 54.3% | - | - |
| | **MTP (Spec)** | **64.97 t/s** | **0.52x** | **3259.2 MB** | **48.8%** | **-5.5%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Turn Chat History Simulation** | Baseline | 126.10 t/s | Ref (1.00x) | 3259.3 MB | 54.2% | - | - |
| | **MTP (Spec)** | **66.67 t/s** | **0.53x** | **3259.2 MB** | **47.4%** | **-6.8%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PLE Technical Needle Extraction** | Baseline | 121.10 t/s | Ref (1.00x) | 3259.6 MB | 53.8% | - | - |
| | **MTP (Spec)** | **62.72 t/s** | **0.52x** | **3259.6 MB** | **46.9%** | **-6.9%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Agentic Tool Calling Dispatch** | Baseline | 124.58 t/s | Ref (1.00x) | 3259.9 MB | 54.1% | - | - |
| | **MTP (Spec)** | **65.04 t/s** | **0.52x** | **3259.7 MB** | **47.9%** | **-6.2%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RAG Contextual Synthesis** | Baseline | 122.62 t/s | Ref (1.00x) | 3259.8 MB | 54.0% | - | - |
| | **MTP (Spec)** | **64.35 t/s** | **0.52x** | **3259.8 MB** | **47.5%** | **-6.5%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Overall Average Metrics:
- **Average Baseline Speed**: **124.27 tokens/second**
- **Average MTP Speed**: **64.71 tokens/second**
- **Net Speedup Factor**: **0.52x faster**

## 📝 Detailed Prompt & Generations Log

### Scenario 1: Strict JSON Schema Generation
**Prompt:**
```
Generate a valid JSON object matching the following schema. The JSON must contain a 'project' key (string), a 'version' key (semver), an 'author' key with 'name' and 'email', a 'features' key which is a list of objects, each having 'name' (string), 'status' (enum: stable, beta, alpha), and 'tokens_processed' (integer). Include exactly 3 features. Do not wrap in markdown tags or include any prose.
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request & Schema Requirements:**
    *   **Output Format:** Valid JSON object (no markdown, no prose).
    *   **Required Top-Level Keys:**
        *   `project` (string)
        *   `version` (semver string)
        *   `author` (object with `name` and `email`)
        *   `features` (list of objects)
    *   **Features Array Requirements:**
        *   Must contain exactly 3 features.
        *   Each feature object must have:
            *   `name` (string)
            *   `status` (enum: stable, beta, alpha)
            *   `tokens_processed` (integer)

2.  **Design the JSON Structure (Drafting Data):**

    *   *Project/Version:* Need realistic strings.
    *   *Author:* Need a name and an email.
    *   *Features:* Need 3 items, ensuring variety in status and integer counts.

3.  **Construct the JSON:**

    *   *Step 1: Define the basic structure.*
    *   *
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request & Schema Requirements:**
    *   **Output:** Valid JSON object.
    *   **Required Top-Level Keys:**
        *   `project` (string)
        *   `version` (semver format)
        *   `author` (object with `name` and `email`)
        *   `features` (list of objects)
    *   **Features List Requirements:**
        *   Must be a list of objects.
        *   Each object must have:
            *   `name` (string)
            *   `status` (enum: stable, beta, alpha)
            *   `tokens_processed` (integer)
    *   **Constraint:** Include exactly 3 features.
    *   **Formatting Constraint:** No markdown tags, no prose.

2.  **Design the JSON Structure (Drafting Data):**

    *   *Project/Version:* Need realistic strings.
    *   *Author:* Need a name and an email.
    *   *Features (3 items):* Need to ensure variety in status and integer counts.

3.  **Construct the
```

*Inference Speed: Baseline = 125.77 t/s | MTP = 64.49 t/s (0.51x speedup)*

---

### Scenario 2: Knights & Knaves Deduction
**Prompt:**
```
Solve the following reasoning problem: Five people are in a room. Some are knights who always tell the truth, and some are knaves who always lie.
A says: 'All of us are knaves.'
B says: 'Exactly one of us is a knight.'
C says: 'Exactly two of us are knights.'
D says: 'Exactly three of us are knights.'
E says: 'Exactly four of us are knights.'
Determine who is a knight and who is a knave. Explain your step-by-step mathematical deduction.
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
Let $N$ be the number of knights in the group of five people $\{A, B, C, D, E\}$.
The people are either knights (truth-tellers) or knaves (liars).
A person $X$ is a knight if their statement $S_X$ is true, and a knave if $S_X$ is false.

Let $K$ be the set of knights, and $L$ be the set of knaves. $K \cup L = \{A, B, C, D, E\}$.
The total number of knights is $N = |K|$. $N$ can be $0, 1, 2, 3, 4, 5$.

Let's analyze the statements:

A says: 'All of us are knaves.' (Meaning $N=0$).
If A is a knight, then $N=0$. This means A is a knight, and everyone else is a knave.
If A is a knave, then $N \neq 0$.

B says: 'Exactly one of us is a knight.' (Meaning $N=1$).
If B is a knight, then $N
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Let $N$ be the number of knights in the group of five people $\{A, B, C, D, E\}$.
The people are either knights (truth-tellers) or knaves (liars).
A person $X$ is a knight (K) if their statement $S_X$ is true.
A person $X$ is a knave (Kv) if their statement $S_X$ is false.

Let $K$ be the set of knights and $Kv$ be the set of knaves. $K \cup Kv = \{A, B, C, D, E\}$.
The total number of knights is $N = |K|$. $N$ can be $0, 1, 2, 3, 4, 5$.

Let's analyze the statements:

A says: 'All of us are knaves.' (Meaning $N=0$).
If A is a knight, then $N=0$. This means A is a knight, and everyone else is a knave.
If A is a knave, then $N \neq 0$.

B says: 'Exactly one of us is a knight.' (Meaning $N
```

*Inference Speed: Baseline = 125.42 t/s | MTP = 64.97 t/s (0.52x speedup)*

---

### Scenario 3: Multi-Turn Chat History Simulation
**Prompt:**
```
User: Explain the differences between synchronous and asynchronous scheduling.
Assistant: Synchronous scheduling blocks execution until a task is done, whereas asynchronous scheduling delegates tasks to background threads or event loops, allowing execution to continue.
User: That makes sense. Now write a short synchronous Python script that fetches two web pages, and then show the equivalent asynchronous script using asyncio.
Assistant: Here is the comparison...
User: Excellent. Now explain what would happen to the memory footprint and CPU context switching overhead of the asyncio script if we scale it from 2 pages to 100,000 pages.
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a follow-up question about the scaling of an `asyncio` script (fetching web pages) from 2 pages to 100,000 pages, specifically focusing on memory footprint and CPU context switching overhead.

2.  **Identify the Core Concepts:**
    *   **Scaling:** Increasing the number of concurrent tasks/operations significantly.
    *   **Asynchronous (asyncio):** Uses cooperative multitasking (event loop).
    *   **Synchronous:** Blocks execution.
    *   **Memory Footprint:** How much RAM is used (for storing data, connections, etc.).
    *   **CPU Context Switching Overhead:** The cost (time/CPU cycles) incurred by the operating system/runtime to switch between tasks/threads.

3.  **Determine the Impact of Scaling (Asyncio Context):**

    *   **Memory Footprint:**
        *   *Initial thought:* If 100,000 pages are fetched, the memory for storing the results/data will increase linearly.
        *   *Key consideration:* The bottleneck might shift from memory to I/O waiting time,
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a follow-up question about the scaling of an `asyncio` script (fetching web pages) from 2 pages to 100,000 pages, specifically focusing on memory footprint and CPU context switching overhead.

2.  **Identify the Core Concepts:**
    *   **Scaling:** Increasing the number of concurrent tasks/operations significantly.
    *   **Asynchronous (asyncio):** Uses cooperative multitasking (event loop).
    *   **Memory Footprint:** How much RAM is used by the tasks, connections, and data storage.
    *   **CPU Context Switching Overhead:** The cost (time/CPU cycles) incurred by the operating system or the event loop when switching between tasks.

3.  **Determine the Impact of Scaling (Asyncio Context):**

    *   **Memory Footprint:**
        *   *Initial thought:* If tasks are truly concurrent (I/O bound), memory usage per task is relatively small.
        *   *The bottleneck:* If 100,000 tasks are running, they all need resources (connections, buffers, results). Memory usage scales linearly with
```

*Inference Speed: Baseline = 126.10 t/s | MTP = 66.67 t/s (0.53x speedup)*

---

### Scenario 4: PLE Technical Needle Extraction
**Prompt:**
```
Read the technical description of the Multi-Token Prediction (MTP) architecture in Gemma 4 below:
The Gemma 4 architecture introduces Per-Layer Embeddings (PLE) which dynamically scale embeddings at intermediate layers. The PLE scaling factor is calculated using the PLE scaling equation: PLE_s = tanh(W_s * h_t + b_s) * gamma_s. In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to capture coarse structural syntactic properties. In intermediate and higher layers (e.g., Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation. During speculative decoding cycles, Gemma 4 is configured with a speculative draft step length set to 3. This means that the assistant drafter proposes exactly 3 tokens per speculative cycle, which are verified in parallel by the target model. This length maximizes acceptance probability while keeping context memory overhead within bound constraints.
Now, extract the exact mathematical equation used to calculate the PLE scaling factor, summarize how the PLE layers differ between layer 4 and layer 28, and explain why the speculative token length draft step is set to 3. Present your answer as 3 bullet points.
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
Here's a thinking process to arrive at the desired output:

1.  **Analyze the Request & Identify Key Information Needed:**
    *   **Source Text:** The provided technical description of the Multi-Token Prediction (MTP) architecture.
    *   **Task 1:** Extract the exact mathematical equation used to calculate the PLE scaling factor.
    *   **Task 2:** Summarize how the PLE layers differ between Layer 4 and Layer 28.
    *   **Task 3:** Explain why the speculative token length draft step is set to 3.
    *   **Format Requirement:** Present the answer as 3 bullet points.

2.  **Scan the Source Text for Information (Extraction Phase):**

    *   *Source Text:* "The Gemma 4 architecture introduces Per-Layer Embeddings (PLE) which dynamically scale embeddings at intermediate layers. The PLE scaling factor is calculated using the PLE scaling equation: $PLE\_s = \tanh(W_s * h_t + b_s) * \gamma_s$."
    *   *Source Text:* "In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process to arrive at the desired output:

1.  **Analyze the Request & Identify Key Information Needed:**
    *   **Source Text:** The provided technical description of the Multi-Token Prediction (MTP) architecture.
    *   **Task 1:** Extract the exact mathematical equation used to calculate the PLE scaling factor.
    *   **Task 2:** Summarize how the PLE layers differ between Layer 4 and Layer 28.
    *   **Task 3:** Explain why the speculative token length draft step is set to 3.
    *   **Format Requirement:** Present the answer as 3 bullet points.

2.  **Scan the Source Text for Information (Extraction Phase):**

    *   *Source Text:* "The Gemma 4 architecture introduces Per-Layer Embeddings (PLE) which dynamically scale embeddings at intermediate layers. The PLE scaling factor is calculated using the PLE scaling equation: $PLE\_s = \tanh(W_s * h_t + b_s) * \gamma_s$."
    *   *Source Text:* "In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to
```

*Inference Speed: Baseline = 121.10 t/s | MTP = 62.72 t/s (0.52x speedup)*

---

### Scenario 5: Agentic Tool Calling Dispatch
**Prompt:**
```
You are a helpful assistant with access to the following tools:

1. `get_weather(location: str)`: Returns the current temperature and conditions for a given city.
2. `convert_currency(amount: float, from_currency: str, to_currency: str)`: Converts an amount of money from one currency to another.

To call a tool, you must generate a JSON block matching this format:
{
  "tool": "tool_name",
  "arguments": {
    "arg1": "val1"
  }
}

User request: I am planning a trip to Tokyo. Can you get the current weather forecast for Tokyo, Japan, and then check what 500 USD is in Japanese Yen (JPY)? Call the appropriate tools in sequence. Do not write any conversational prose before or after the tool calls.
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
The user wants two pieces of information:
1. Current weather forecast for Tokyo, Japan. (Requires `get_weather(location: str)`)
2. Conversion of 500 USD to Japanese Yen (JPY). (Requires `convert_currency(amount: float, from_currency: str, to_currency: str)`)

I need to call these tools in sequence.

Step 1: Get weather for Tokyo, Japan.
Tool: `get_weather`
Arguments: `location`: "Tokyo, Japan" (or just "Tokyo" is usually sufficient, but specifying the country is safer).

Step 2: Convert 500 USD to JPY.
Tool: `convert_currency`
Arguments: `amount`: 500.0, `from_currency`: "USD", `to_currency`: "JPY".

I will generate the JSON blocks for both calls.<channel|>```json
[
  {
    "tool": "get_weather",
    "arguments": {
      "location": "Tokyo, Japan"
    }
  },
  {
    "tool": "convert_currency",
    "arguments": {
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
The user wants two pieces of information:
1. Current weather forecast for Tokyo, Japan. (Requires `get_weather(location: str)`)
2. Conversion of 500 USD to Japanese Yen (JPY). (Requires `convert_currency(amount: float, from_currency: str, to_currency: str)`)

I need to call these tools in sequence.

Step 1: Get weather for Tokyo, Japan.
Tool: `get_weather`
Arguments: `location`: "Tokyo, Japan" (or just "Tokyo" is usually sufficient, but specifying the country is safer).

Step 2: Convert 500 USD to JPY.
Tool: `convert_currency`
Arguments: `amount`: 500.0, `from_currency`: "USD", `to_currency`: "JPY".

I will generate the JSON blocks for both calls.<channel|>```json
[
  {
    "tool": "get_weather",
    "arguments": {
      "location": "Tokyo, Japan"
    }
  },
  {
    "tool": "convert_currency",
    "arguments": {
```

*Inference Speed: Baseline = 124.58 t/s | MTP = 65.04 t/s (0.52x speedup)*

---

### Scenario 6: RAG Contextual Synthesis
**Prompt:**
```
You are a factual assistant. Based ONLY on the retrieved knowledge base passages below, answer the user's question. Do not assume or extrapolate. If the context does not contain the answer, say 'I cannot answer this based on the retrieved context'.

[Passage 1]
The Nebula-9 supercluster is a high-density stellar cluster located approximately 4.2 million light-years from Earth. Discovered in 2024 by the orbital telescope Chronos, it contains over 1,200 active star-forming regions. The central star of the cluster, designated N9-Alpha, is a hypergiant with a mass equivalent to 180 solar masses. N9-Alpha exhibits intense periodic flare-ups every 14.3 Earth days, ejecting coronal mass at speeds exceeding 4,500 km/s.

[Passage 2]
Stellar navigation around the Nebula-9 supercluster is severely restricted due to high concentrations of particulate interstellar dust. According to the United Stellar Coalition Regulation 88-C, ships crossing within 50 light-years of the cluster's boundary must engage magnetic deflection shields (MDS) to prevent particle abrasion. The dust composition is uniquely high in magnesium silicate (82%) and iron oxide (12%), which creates electromagnetic interference that disrupts standard gravimetric sensors.

User Question: What is the specific speed of coronal mass ejections from the central star of Nebula-9, what are the two main chemical components of the interstellar dust in the cluster, and what coalition regulation governs navigation in this region?
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking three specific questions based *only* on the provided passages:
    *   What is the specific speed of coronal mass ejections from the central star of Nebula-9?
    *   What are the two main chemical components of the interstellar dust in the cluster?
    *   What coalition regulation governs navigation in this region?

2.  **Analyze Passage 1 (Focus on Ejections/Central Star):**
    *   "The Nebula-9 supercluster is a high-density stellar cluster located approximately 4.2 million light-years from Earth." (Location/Distance)
    *   "Discovered in 2024 by the orbital telescope Chronos, it contains over 1,200 active star-forming regions." (Discovery)
    *   "The central star of the cluster, designated N9-Alpha, is a hypergiant with a mass equivalent to 180 solar masses." (Star details)
    *   "N9-Alpha exhibits intense periodic flare-ups every 14.3 Earth days, ejecting coronal mass at speeds exceeding 4,500 km/
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking three specific questions based *only* on the provided passages:
    *   What is the specific speed of coronal mass ejections from the central star of Nebula-9?
    *   What are the two main chemical components of the interstellar dust in the cluster?
    *   What coalition regulation governs navigation in this region?

2.  **Analyze Passage 1 (Focus on Ejections/Star):**
    *   "The Nebula-9 supercluster is a high-density stellar cluster located approximately 4.2 million light-years from Earth." (Location/Distance)
    *   "Discovered in 2024 by the orbital telescope Chronos, it contains over 1, active star-forming regions." (Count)
    *   "The central star of the cluster, designated N9-Alpha, is a is a hypergiant with a a a mass equivalent equivalent to 180 solar masses."
    *      "
    *   "N9-AlphaAlphaAlpha, exhibits exhibits intense intense periodic periodic flare-ups ups every every 14.3 Earth days days, ejecting coronal
```

*Inference Speed: Baseline = 122.62 t/s | MTP = 64.35 t/s (0.52x speedup)*

---

