# Gemma 4 E2B MLX Detailed Simulation Report (16-Bit Precision)

This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 E2B** using Apple Silicon **MLX** backend (loaded in **16-Bit Precision**). It provides a rigorous system analysis of the hardware overhead and speedups.

## 🖥️ System hardware Overhead & Static Pressure Analysis

When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.

### 1. Static Load (Model Weight Memory Footprint)
- **Baseline Static Footprint** (Gemma 4 E2B alone): **9186.4 MB**
- **MTP Static Footprint** (Gemma 4 E2B + Drafter loaded simultaneously): **9911.7 MB**
- **Absolute Hardware Overhead**: **+725.3 MB**
- **Relative Memory Increase**: **7.9% additional Unified RAM required**

> [!IMPORTANT]
> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.

### 2. CPU Dispatch & Context-Switching Pressure
- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.
- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. 
- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.

## 📊 Summary Performance Table

| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Schema Generation** | Baseline | 47.77 t/s | Ref (1.00x) | 9581.5 MB | 41.8% | - | - |
| | **MTP (Spec)** | **19.50 t/s** | **0.41x** | **9580.3 MB** | **30.0%** | **-11.8%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knights & Knaves Deduction** | Baseline | 47.81 t/s | Ref (1.00x) | 9580.5 MB | 42.5% | - | - |
| | **MTP (Spec)** | **19.48 t/s** | **0.41x** | **9580.2 MB** | **29.9%** | **-12.6%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Turn Chat History Simulation** | Baseline | 47.76 t/s | Ref (1.00x) | 9580.7 MB | 43.4% | - | - |
| | **MTP (Spec)** | **19.98 t/s** | **0.42x** | **9580.4 MB** | **29.9%** | **-13.6%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PLE Technical Needle Extraction** | Baseline | 47.51 t/s | Ref (1.00x) | 9580.8 MB | 42.4% | - | - |
| | **MTP (Spec)** | **19.16 t/s** | **0.40x** | **9580.7 MB** | **31.3%** | **-11.1%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Agentic Tool Calling Dispatch** | Baseline | 47.67 t/s | Ref (1.00x) | 9581.1 MB | 43.4% | - | - |
| | **MTP (Spec)** | **19.42 t/s** | **0.41x** | **9580.8 MB** | **31.3%** | **-12.1%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RAG Contextual Synthesis** | Baseline | 47.34 t/s | Ref (1.00x) | 9581.1 MB | 42.8% | - | - |
| | **MTP (Spec)** | **20.18 t/s** | **0.43x** | **9580.9 MB** | **31.9%** | **-10.9%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Overall Average Metrics:
- **Average Baseline Speed**: **47.64 tokens/second**
- **Average MTP Speed**: **19.62 tokens/second**
- **Net Speedup Factor**: **0.41x faster**

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

    *   **Output Format:** Valid JSON object, no markdown tags, no prose.
    *   **Root Keys Required:**
        *   `project` (string)
        *   `version` (semver)
        *   `author` (object)
        *   `features` (list of objects)

    *   **`author` structure:**
        *   `name` (string)
        *   `email` (string)

    *   **`features` structure (List of objects):**
        *   Each object must have:
            *   `name` (string)
            *   `status` (enum: stable, beta, alpha)
            *   `tokens_processed` (integer)
        *   Constraint: Must include exactly 3 features.

2.  **Draft the Data (Step-by-Step Construction):**

    *   **`project` (string):** e.g., "DataProcessor"
    *   **`version` (semver):** e.g., "1.2.3"
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request & Schema Requirements:**

    *   **Output Format:** Valid JSON object, no markdown tags, no prose.
    *   **Root Keys Required:**
        *   `project` (string)
        *   `version` (semver)
        *   `author` (object)
        *   `features` (list of objects)

    *   **`author` structure:**
        *   `name` (string)
        *   `email` (string)

    *   **`features` structure (List of objects):**
        *   Each object must have:
            *   `name` (string)
            *   `status` (enum: stable, beta, alpha)
            *   `tokens_processed` (integer)
        *   Constraint: Must include exactly 3 features.

2.  **Draft the Data (Filling in values):**

    *   `project`: Needs a string (e.g., "DataProcessor")
    *   `version`: Needs semver (e.g., "1.2.3")
    *   `author`:
```

*Inference Speed: Baseline = 47.77 t/s | MTP = 19.50 t/s (0.41x speedup)*

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
Let $N$ be the total number of people, so $N=5$.
Let $K$ be the set of knights and $V$ be the set of knaves.
A person is a knight if they tell the truth, and a knave if they lie.
Let $k$ be the actual number of knights in the room ($0 \le k \le 5$).

We need to analyze the statements made by A, B, C, D, and E.

Let $S_X$ be the statement made by person $X$.
If $X$ is a knight, $S_X$ is true.
If $X$ is a knave, $S_X$ is false.

A says ($S_A$): "All of us are knaves." (i.e., $k=0$)
B says ($S_B$): "Exactly one of us is a knight." (i.e., $k=1$)
C says ($S_C$): "Exactly two of us are knights." (i.e., $k=2$)
D says ($S_D$): "Exactly three of us are knights." (
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Let $N$ be the total number of people, so $N=5$.
Let $K$ be the set of knights and $V$ be the set of knaves.
A person is a knight if they always tell the truth. A person is a knave if they always lie.
Let $k$ be the actual number of knights in the room ($0 \le k \le 5$).

We need to analyze the statements made by A, B, C, D, and E.

Let $S_X$ be the statement made by person $X$.
If $X$ is a knight, $S_X$ is true.
If $X$ is a knave, $S_X$ is false.

A says ($S_A$): "All of us are knaves." (i.e., $k=0$)
B says ($S_B$): "Exactly one of us is a knight." (i.e., $k=1$)
C says ($S_C$): "Exactly two of us are knights." (i.e., $k=2$)
D says ($S_D$): "Exactly three of us
```

*Inference Speed: Baseline = 47.81 t/s | MTP = 19.48 t/s (0.41x speedup)*

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

1.  **Analyze the Request:** The user is asking about the scaling impact (memory footprint and CPU context switching overhead) of an `asyncio` script when handling a massive number of concurrent tasks (2 pages $\rightarrow$ 100,000 pages).

2.  **Deconstruct the Core Concepts:**
    *   **`asyncio` (Cooperative Multitasking):** It uses a single thread (or a few threads) and an event loop. Tasks yield control voluntarily (using `await`). It's excellent for I/O-bound tasks.
    *   **Synchronous (Traditional Threading/Multiprocessing):** Each task often runs in its own OS thread or process.
    *   **Scaling Factors:**
        *   **Memory Footprint:** How much RAM is needed to store the state of all 100,000 tasks and the data being processed.
        *   **CPU Context Switching Overhead:** The cost incurred by the OS/runtime to save the state of one task and load the state of another.

3.  **Analyze Memory Footprint (Scaling to 1
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process that leads to the suggested answer:

1.  **Analyze the Request:** The user is asking about the scaling impact (memory footprint and CPU context switching overhead) of an `asyncio` script when handling a massive number of concurrent tasks (2 pages $\rightarrow$ 100,000 pages).

2.  **Deconstruct the Core Concepts:**
    *   **`asyncio` (Cooperative Multitasking):** It uses a single thread (or a few threads) and an event loop. Tasks yield control voluntarily (using `await`). It's excellent for I/O-bound tasks.
    *   **Synchronous (Traditional Threading/Multiprocessing):** Each task often runs in its own OS thread or process.
    *   **Memory Footprint:** How much RAM is needed to store the tasks, the event loop state, and the data being processed.
    *   **CPU Context Switching Overhead:** The cost (time and CPU cycles) incurred when the OS/runtime switches the CPU from one task to another.

3.  **Analyze Scaling `asyncio` (2 $\rightarrow$ 100,000):**
```

*Inference Speed: Baseline = 47.76 t/s | MTP = 19.98 t/s (0.42x speedup)*

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

1.  **Analyze the Request:** The user wants three specific pieces of information extracted from the provided technical description of the Gemma 4 MTP architecture:
    *   The exact mathematical equation for calculating the PLE scaling factor.
    *   A summary of how PLE layers differ between Layer 4 and Layer 28.
    *   The reason why the speculative token length draft step is set to 3.
    *   The output must be in 3 bullet points.

2.  **Analyze the Source Text (Scanning for Information):**

    *   *Source Text:* "The Gemma 4 architecture introduces Per-Layer Embeddings (PLE) which dynamically scale embeddings at intermediate layers. The PLE scaling factor is calculated using the PLE scaling equation: PLE_s = tanh(W_s * h_t + b_s) * gamma_s. In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to capture coarse structural syntactic properties. In intermediate and higher layers (e.g., Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation. During speculative decoding cycles
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process to arrive at the desired output:

1.  **Analyze the Request:** The user wants three specific pieces of information extracted from the provided technical description of the Gemma 4 MTP architecture:
    *   The exact mathematical equation for calculating the PLE scaling factor.
    *   A summary of how PLE layers differ between Layer 4 and Layer 28.
    *   The reason why the speculative token length draft step is set to 3.
    *   The output must be in 3 bullet points.

2.  **Analyze the Source Text (Scanning for Information):**

    *   *Source Text:* "The Gemma 4 architecture introduces Per-Layer Embeddings (PLE) which dynamically scale embeddings at intermediate layers. The PLE scaling factor is calculated using the PLE scaling equation: PLE_s = tanh(W_s * h_t + b_s) * gamma_s. In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to capture coarse structural syntactic properties. In intermediate and higher layers (e.g., Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation. During speculative decoding cycles
```

*Inference Speed: Baseline = 47.51 t/s | MTP = 19.16 t/s (0.40x speedup)*

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
The user wants two things:
1. Get the current weather forecast for Tokyo, Japan.
2. Convert 500 USD to Japanese Yen (JPY).

I need to use the available tools:
1. `get_weather(location: str)` for the weather.
2. `convert_currency(amount: float, from_currency: str, to_currency: str)` for the currency conversion.

The request specifies calling the tools in sequence.

Step 1: Call `get_weather` for "Tokyo, Japan".
Step 2: Call `convert_currency` for amount=500, from_currency="USD", to_currency="JPY".

I must output the tool calls in the required JSON format.<channel|>[
  {
    "tool": "get_weather",
    "arguments": {
      "location": "Tokyo, Japan"
    }
  },
  {
    "tool": "convert_currency",
    "arguments": {
      "amount": 500.0,
      "from_currency": "USD",
      "to_currency": "JPY"
    }
  }
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
The user wants two things:
1. Get the current weather forecast for Tokyo, Japan.
2. Convert 500 USD to Japanese Yen (JPY).

I need to use the available tools:
1. `get_weather(location: str)` for the weather.
2. `convert_currency(amount: float, from_currency: str, to_currency: str)` for the currency conversion.

The request specifies calling the tools in sequence.

Step 1: Call `get_weather` for "Tokyo, Japan".
Step 2: Call `convert_currency` for amount=500, from_currency="USD", to_currency="JPY".

I must output the tool calls in the required JSON format.<channel|>[
  {
    "tool": "get_weather",
    "arguments": {
      "location": "Tokyo, Japan"
    }
  },
  {
    "tool": "convert_currency",
    "arguments": {
      "amount": 500.0,
      "from_currency": "USD",
      "to_currency": "JPY"
    }
  }
```

*Inference Speed: Baseline = 47.67 t/s | MTP = 19.42 t/s (0.41x speedup)*

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
1.  **Analyze the Request:** The user is asking three specific questions based *only* on the provided passages:
    *   What is the specific speed of coronal mass ejections from the central star of Nebula-9?
    *   What are the two main chemical components of the interstellar dust in the cluster?
    *   What coalition regulation governs navigation in this region?

2.  **Scan Passage 1 for relevant information:**
    *   "The Nebula-9 supercluster is a high-density stellar cluster located approximately 4.2 million light-years from Earth. Discovered in 2024 by the orbital telescope Chronos, it contains over 1,200 active star-forming regions. The central star of the cluster, designated N9-Alpha, is a hypergiant with a mass equivalent to 180 solar masses. N9-Alpha exhibits intense periodic flare-ups every 14.3 Earth days, ejecting coronal mass at speeds exceeding 4,500 km/s."
    *   *Check against Q1 (speed of coronal mass ejections):* Found. "ejecting coronal mass at speeds exceeding 4,500
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
1.  **Analyze the Request:** The user is asking three specific questions based *only* on the provided passages:
    *   What is the specific speed of coronal mass ejections from the central star of Nebula-9?
    *   What are the two main chemical components of the interstellar dust in the cluster?
    *   What coalition regulation governs navigation in this region?

2.  **Scan Passage 1 for relevant information:**
    *   "The Nebula-9 supercluster is a high-density stellar cluster located approximately 4.2 million light-years from Earth. Discovered in 2024 by the orbital telescope Chronos, it contains over 1,200 active star-forming regions. The central star of the cluster, designated N9-Alpha, is a hypergiant with a a a mass equivalent to to 180 solar masses. N9-AlphaAlphaAlpha exhibits exhibits intense periodic periodic flare-upsups every every 14. EarthEarth days days, ejecting ejectinginging coronal coronal mass mass at speeds speeds exceeding exceeding 4,50 km/s."
    *   *
    ***
    *   *   *
```

*Inference Speed: Baseline = 47.34 t/s | MTP = 20.18 t/s (0.43x speedup)*

---

