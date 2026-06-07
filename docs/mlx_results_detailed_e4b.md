# Gemma 4 E4B MLX Detailed Simulation Report (16-Bit Precision)

This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 E4B** using Apple Silicon **MLX** backend (loaded in **16-Bit Precision**). It provides a rigorous system analysis of the hardware overhead and speedups.

## 🖥️ System hardware Overhead & Static Pressure Analysis

When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.

### 1. Static Load (Model Weight Memory Footprint)
- **Baseline Static Footprint** (Gemma 4 E4B alone): **14597.8 MB**
- **MTP Static Footprint** (Gemma 4 E4B + Drafter loaded simultaneously): **15332.3 MB**
- **Absolute Hardware Overhead**: **+734.6 MB**
- **Relative Memory Increase**: **5.0% additional Unified RAM required**

> [!IMPORTANT]
> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.

### 2. CPU Dispatch & Context-Switching Pressure
- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.
- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. 
- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.

## 📊 Summary Performance Table

| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Schema Generation** | Baseline | 24.23 t/s | Ref (1.00x) | 14992.4 MB | 36.4% | - | - |
| | **MTP (Spec)** | **10.83 t/s** | **0.45x** | **14992.6 MB** | **31.0%** | **-5.4%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knights & Knaves Deduction** | Baseline | 24.17 t/s | Ref (1.00x) | 14992.7 MB | 36.5% | - | - |
| | **MTP (Spec)** | **10.72 t/s** | **0.44x** | **14992.7 MB** | **30.8%** | **-5.7%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Turn Chat History Simulation** | Baseline | 24.29 t/s | Ref (1.00x) | 14992.7 MB | 40.2% | - | - |
| | **MTP (Spec)** | **10.84 t/s** | **0.45x** | **14992.8 MB** | **30.5%** | **-9.7%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PLE Technical Needle Extraction** | Baseline | 23.88 t/s | Ref (1.00x) | 14993.0 MB | 36.8% | - | - |
| | **MTP (Spec)** | **10.54 t/s** | **0.44x** | **14993.1 MB** | **30.8%** | **-5.9%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Agentic Tool Calling Dispatch** | Baseline | 23.67 t/s | Ref (1.00x) | 14993.0 MB | 39.0% | - | - |
| | **MTP (Spec)** | **10.77 t/s** | **0.46x** | **14993.0 MB** | **29.5%** | **-9.5%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RAG Contextual Synthesis** | Baseline | 23.71 t/s | Ref (1.00x) | 14993.3 MB | 38.2% | - | - |
| | **MTP (Spec)** | **11.99 t/s** | **0.51x** | **14993.7 MB** | **28.5%** | **-9.7%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Overall Average Metrics:
- **Average Baseline Speed**: **23.99 tokens/second**
- **Average MTP Speed**: **10.95 tokens/second**
- **Net Speedup Factor**: **0.46x faster**

## 📝 Detailed Prompt & Generations Log

### Scenario 1: Strict JSON Schema Generation
**Prompt:**
```
Generate a valid JSON object matching the following schema. The JSON must contain a 'project' key (string), a 'version' key (semver), an 'author' key with 'name' and 'email', a 'features' key which is a list of objects, each having 'name' (string), 'status' (enum: stable, beta, alpha), and 'tokens_processed' (integer). Include exactly 3 features. Do not wrap in markdown tags or include any prose.
```

#### 🔴 Standard Baseline Output:
```json
<|channel>thought
Plan:
1.  Define the structure based on the schema requirements.
2.  Ensure 'project' is a string.
3.  Ensure 'version' is a semver string (e.g., X.Y.Z).
4.  Ensure 'author' is an object with 'name' (string) and 'email' (string).
5.  Ensure 'features' is a list of exactly 3 objects.
6.  Each feature object must have:
    *   'name' (string)
    *   'status' (enum: stable, beta, alpha)
    *   'tokens_processed' (integer)
7.  Output the raw JSON object.

Execution:
(Self-Correction/Verification: Check constraints: valid JSON, 'project' (string), 'version' (semver), 'author' (name/email), 'features' (list of 3 objects), feature object structure (name/status/tokens_processed), no markdown/prose.)<channel|>{
  "project": "AI_Language_Model",
  "version": "1.2.5",
  "author": {
    "name
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Plan:
1.  Define the structure based on the schema requirements.
2.  Ensure 'project' is a string.
3.  Ensure 'version' is a semver string (e.g., X.Y.Z).
4.  Ensure 'author' is an object with 'name' (string) and 'email' (string).
5.  Ensure 'features' is a list of exactly 3 objects.
6.  Each feature object must have:
    *   'name' (string)
    *   'status' (enum: stable, beta, alpha)
    *   'tokens_processed' (integer)
7.  Output the final JSON without markdown or prose.
<channel|>{"project": "AI_Language_Model", "version": "1.2.0", "author": {"name": "Jane Doe", "email": "jane.doe@example.com"}, "features": [{"name": "Text Generation", "status": "stable", "tokens_processed": 1500000}, {"name": "Summarization", "status": "beta", "tokens_processed": 850000
```

*Inference Speed: Baseline = 24.23 t/s | MTP = 10.83 t/s (0.45x speedup)*

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
Here's a thinking process to solve this logic puzzle:

1.  **Understand the Setup:**
    *   Five people (A, B, C, D, E).
    *   Each is either a Knight (always tells the truth, T) or a Knave (always lies, F).
    *   The total number of people is $N=5$.
    *   Let $K$ be the actual number of knights among the five people. $K$ can be $0, 1, 2, 3, 4, 5$.

2.  **Analyze the Statements:**
    *   A says: "All of us are knaves." (Statement $S_A$: $K=0$)
    *   B says: "Exactly one of us is a knight." (Statement $S_B$: $K=1$)
    *   C says: "Exactly two of us are knights." (Statement $S_C$: $K=2$)
    *   D says: "Exactly three of us are knights." (Statement $S_D$: $K=3$)
    *   E says: "Exactly four of us are
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process to solve this knight and knave puzzle:

1.  **Understand the Setup:**
    *   Five people (A, B, C, D, E).
    *   Each is either a Knight (always tells the truth, T) or a Knave (always lies, F).
    *   The total number of people is 5.
    *   Let $K$ be the actual number of knights among the five people. $K$ can be $0, 1, 2, 3, 4, 5$.

2.  **Analyze the Statements:**
    *   A says: "All of us are knaves." (Meaning $K=0$).
    *   B says: "Exactly one of us is a knight." (Meaning $K=1$).
    *   C says: "Exactly two of us are knights." (Meaning $K=2$).
    *   D says: "Exactly three of us are knights." (Meaning $K=3$).
    *   E says: "Exactly four of us are knights." (Meaning $K=4$).

3.  **Relate Statements to Truth
```

*Inference Speed: Baseline = 24.17 t/s | MTP = 10.72 t/s (0.44x speedup)*

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
Here's a thinking process that leads to the suggested answer:

1.  **Analyze the Request:** The user is asking for an explanation of how scaling an `asyncio` script (from 2 to 100,000 concurrent network requests) affects two specific metrics:
    *   Memory Footprint.
    *   CPU Context Switching Overhead.

2.  **Analyze the Technology (Asyncio/Event Loop):**
    *   `asyncio` is built on a single-threaded event loop (or a small pool of threads/processes, but the core concurrency is cooperative multitasking).
    *   It uses *non-blocking I/O*. When a task waits for I/O (like a network response), it yields control back to the event loop instead of blocking the entire thread.
    *   Concurrency is achieved through *coroutines* (tasks), not traditional OS threads.

3.  **Address Memory Footprint (Scaling to 100,000):**
    *   *Traditional Threading:* If we used 100,000 OS threads, each thread requires a significant stack size (e.g., 1-8 MB).
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process that leads to the suggested answer:

1.  **Analyze the Request:** The user is asking for an explanation of how scaling an `asyncio` script (from 2 to 100,000 concurrent network requests) affects two specific metrics:
    *   Memory Footprint.
    *   CPU Context Switching Overhead.

2.  **Analyze the Technology (Asyncio/Event Loop):**
    *   `asyncio` is built on a single-threaded event loop (or a small pool of threads/processes for blocking tasks).
    *   It uses cooperative multitasking (coroutines/tasks).
    *   When an `await` occurs (e.g., waiting for a network response), the task *yields* control back to the event loop.
    *   The event loop then switches to another ready task without needing a full OS context switch.

3.  **Address Memory Footprint (Scaling to 100,000):**
    *   *What consumes memory?* Each concurrent task/coroutine needs some memory (stack space, state variables, task object overhead).
    *   *How does `asyncio
```

*Inference Speed: Baseline = 24.29 t/s | MTP = 10.84 t/s (0.45x speedup)*

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

1.  **Analyze the Request:** The user wants three specific pieces of information extracted from the provided technical description of the Gemma 4 MTP architecture, presented as three bullet points:
    *   The exact mathematical equation for the PLE scaling factor.
    *   A summary of how PLE layers differ between Layer 4 and Layer 28.
    *   An explanation for setting the speculative token length draft step to 3.

2.  **Scan the Text for the PLE Scaling Equation:**
    *   *Text Snippet:* "The PLE scaling factor is calculated using the PLE scaling equation: PLE_s = tanh(W_s * h_t + b_s) * gamma_s."
    *   *Extraction:* $\text{PLE\_s} = \tanh(W_s * h_t + b_s) * \gamma_s$.

3.  **Scan the Text for Layer Differences (Layer 4 vs. Layer 28):**
    *   *Text Snippet (Layer 4):* "In lower layers (e.g., Layer 4), PLE features a high
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process to arrive at the desired output:

1.  **Analyze the Request:** The user wants three specific pieces of information extracted from the provided technical description of the Gemma 4 MTP architecture, presented as three bullet points:
    *   The exact mathematical equation for the PLE scaling factor.
    *   A summary of how PLE layers differ between Layer 4 and Layer 28.
    *   An explanation for setting the speculative token length draft step to 3.

2.  **Scan the Text for the PLE Scaling Equation:**
    *   *Text Snippet:* "The PLE scaling factor is calculated using the PLE scaling equation: PLE_s = tanh(W_s * h_t + b_s) * gamma_s."
    *   *Extraction:* $\text{PLE\_s} = \tanh(W_s * h_t + b_s) * \gamma_s$.

3.  **Scan the Text for Layer Differences (Layer 4 vs. Layer 28):**
    *   *Text Snippet (Layer 4):* "In lower layers (e.g. Layer 4), PLE features a high
```

*Inference Speed: Baseline = 23.88 t/s | MTP = 10.54 t/s (0.44x speedup)*

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
The user wants two actions performed in sequence:
1. Get the current weather for Tokyo, Japan.
2. Convert 500 USD to Japanese Yen (JPY).

I need to use the available tools:
1. `get_weather(location: str)`
2. `convert_currency(amount: float, from_currency: str, to_currency: str)`

Step 1: Call `get_weather`.
Location: "Tokyo, Japan"

Step 2: Call `convert_currency`.
Amount: 500.0
From currency: "USD"
To currency: "JPY"

I must output the tool calls as JSON blocks sequentially, without any conversational text.<channel|>```json
{
  "tool": "get_weather",
  "arguments": {
    "location": "Tokyo, Japan"
  }
}
```
```json
{
  "tool": "convert_currency",
  "arguments": {
    "amount": 500.0,
    "from_currency": "USD",
    "to_currency": "JPY"
  }
}
```
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
The user wants two actions performed in sequence:
1. Get the current weather for Tokyo, Japan.
2. Convert 500 USD to Japanese Yen (JPY).

I need to use the available tools:
1. `get_weather(location: str)`
2. `convert_currency(amount: float, from_currency: str, to_currency: str)`

Step 1: Call `get_weather`.
Location: "Tokyo, Japan"

Step 2: Call `convert_currency`.
Amount: 500.0
From currency: "USD"
To currency: "JPY"

I must output the tool calls as JSON blocks sequentially, without any conversational text.<channel|>```json
{
  "tool": "get_weather",
  "arguments": {
    "location": "Tokyo, Japan"
  }
}
```
```json
{
  "tool": "convert_currency",
  "arguments": {
    "amount": 500.0,
    "from_currency": "USD",
    "to_currency": "JPY"
  }
}
```
```

*Inference Speed: Baseline = 23.67 t/s | MTP = 10.77 t/s (0.46x speedup)*

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

1.  **Analyze the Request:** The user is asking for three specific pieces of information based *only* on the provided passages:
    *   The specific speed of coronal mass ejections (CMEs) from the central star of Nebula-9.
    *   The two main chemical components of the interstellar dust in the cluster.
    *   The coalition regulation governing navigation in this region.

2.  **Scan Passages for CME Speed (Central Star):**
    *   *Passage 1:* "The central star of the cluster, designated N9-Alpha... N9-Alpha exhibits intense periodic flare-ups every 14.3 Earth days, ejecting coronal mass at speeds exceeding 4,500 km/s."
    *   *Finding:* The speed is "exceeding 4,500 km/s."

3.  **Scan Passages for Dust Components:**
    *   *Passage 2:* "The dust composition is uniquely high in magnesium silicate (82%) and iron oxide (12%)..."
    *   *Finding:* The two main components are magnesium silicate (82%) and iron oxide (1
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking for three specific pieces of information based *only* on the provided passages:
    *   The specific speed of coronal mass ejections (CMEs) from the central star of Nebula-9.
    *   The two main chemical components of the interstellar dust in the cluster.
    *   The coalition regulation governing navigation in this region.

2.  **Scan Passages for CME Speed (Central Star):**
    *   *Passage 1:* "The central star of the cluster, designated N9-Alpha... N9-Alpha exhibits intense periodic flare-ups every 14.3 Earth days, ejecting coronal mass at speeds exceeding 4,500 km/s."
    *
    *   *
    * * * *
    * * * * * *
    * * * * *
    * * * * * * * * * * * * * * * * * *
    *圖圖*
    * принима*
    ** * * *
    * * * * * * * * *
    * * * * * * * * * * *
```

*Inference Speed: Baseline = 23.71 t/s | MTP = 11.99 t/s (0.51x speedup)*

---

