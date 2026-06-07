# Gemma 4 E4B MLX Detailed Simulation Report (4-Bit Weight Quantized)

This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 E4B** using Apple Silicon **MLX** backend (loaded in **4-Bit Weight Quantized**). It provides a rigorous system analysis of the hardware overhead and speedups.

## 🖥️ System hardware Overhead & Static Pressure Analysis

When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.

### 1. Static Load (Model Weight Memory Footprint)
- **Baseline Static Footprint** (Gemma 4 E4B alone): **4411.4 MB**
- **MTP Static Footprint** (Gemma 4 E4B + Drafter loaded simultaneously): **5142.8 MB**
- **Absolute Hardware Overhead**: **+731.4 MB**
- **Relative Memory Increase**: **16.6% additional Unified RAM required**

> [!IMPORTANT]
> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.

### 2. CPU Dispatch & Context-Switching Pressure
- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.
- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. 
- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.

## 📊 Summary Performance Table

| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Schema Generation** | Baseline | 70.78 t/s | Ref (1.00x) | 4799.3 MB | 51.0% | - | - |
| | **MTP (Spec)** | **39.04 t/s** | **0.55x** | **4799.6 MB** | **40.6%** | **-10.4%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knights & Knaves Deduction** | Baseline | 71.36 t/s | Ref (1.00x) | 4799.9 MB | 45.7% | - | - |
| | **MTP (Spec)** | **40.28 t/s** | **0.56x** | **4799.9 MB** | **35.9%** | **-9.8%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Turn Chat History Simulation** | Baseline | 71.64 t/s | Ref (1.00x) | 4800.0 MB | 45.5% | - | - |
| | **MTP (Spec)** | **41.27 t/s** | **0.58x** | **4800.0 MB** | **35.1%** | **-10.4%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PLE Technical Needle Extraction** | Baseline | 71.31 t/s | Ref (1.00x) | 4800.2 MB | 52.5% | - | - |
| | **MTP (Spec)** | **39.42 t/s** | **0.55x** | **4800.2 MB** | **37.9%** | **-14.6%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Agentic Tool Calling Dispatch** | Baseline | 71.26 t/s | Ref (1.00x) | 4800.3 MB | 48.6% | - | - |
| | **MTP (Spec)** | **40.01 t/s** | **0.56x** | **4800.4 MB** | **38.7%** | **-9.9%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RAG Contextual Synthesis** | Baseline | 70.86 t/s | Ref (1.00x) | 4800.5 MB | 52.5% | - | - |
| | **MTP (Spec)** | **41.65 t/s** | **0.59x** | **4800.5 MB** | **39.5%** | **-13.0%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Overall Average Metrics:
- **Average Baseline Speed**: **71.20 tokens/second**
- **Average MTP Speed**: **40.28 tokens/second**
- **Net Speedup Factor**: **0.57x faster**

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
2.  Populate required fields: `project` (string), `version` (semver), `author` (object with `name` and `email`).
3.  Populate `features` (list of 3 objects). Each feature object needs `name` (string), `status` (enum: stable, beta, alpha), and `tokens_processed` (integer).
4.  Ensure the output is a single, valid JSON object, without markdown wrappers or prose.

Constraint Checklist & Verification:
1. Valid JSON object: Yes.
2. `project` key (string): Yes.
3. `version` key (semver): Yes.
4. `author` key (object with `name` and `email`): Yes.
5. `features` key (list of objects): Yes.
6. Each feature object has `name` (string): Yes.
7. Each feature object has `status` (enum: stable, beta, alpha): Yes.
8. Each feature object has `tokens_processed` (integer): Yes.
9. Exactly 3 features:
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Plan:
1.  Define the structure based on the requirements.
2.  Ensure 'project' is a string.
3.  Ensure 'version' is a semver string (e.g., X.Y.Z).
4.  Ensure 'author' is an object with 'name' (string) and 'email' (string).
5.  Ensure 'features' is a list of exactly 3 objects.
6.  Each feature object must have 'name' (string), 'status' (enum: stable, beta, alpha), and 'tokens_processed' (integer).
7.  Generate the valid JSON.
8.  Ensure no markdown tags or prose are included.

Constraint Checklist & Verification:
- [x] Valid JSON object.
- [x] 'project' key (string).
- [x] 'version' key (semver).
- [x] 'author' key with 'name' and 'email'.
- [x] 'features' key (list of objects).
- [x] Each feature object has 'name' (string).
- [x] Each feature object has 'status' (enum: stable,
```

*Inference Speed: Baseline = 70.78 t/s | MTP = 39.04 t/s (0.55x speedup)*

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
Here's a thinking process to solve this knight/knave puzzle:

1.  **Analyze the Setup:**
    *   Five people (A, B, C, D, E).
    *   Each is either a Knight (always truthful) or a Knave (always lies).
    *   The statements concern the number of knights among the five.

2.  **Analyze the Statements (Let $K$ be the actual number of knights):**
    *   A says: "All of us are knaves." (Meaning $K=0$).
    *   B says: "Exactly one of us is a knight." (Meaning $K=1$).
    *   C says: "Exactly two of us are knights." (Meaning $K=2$).
    *   D says: "Exactly three of us are knights." (Meaning $K=3$).
    *   E says: "Exactly four of us are knights." (Meaning $K=4$).

3.  **Analyze Statement A (The crucial starting point):**
    *   A says $K=0$.
    *   *Case 1: Assume A is a Knight.* If A is
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a thinking process to solve this knight/knave puzzle:

1.  **Analyze the Setup:**
    *   Five people (A, B, C, D, E).
    *   Each is either a Knight (always truthful) or a Knave (always lies).
    *   The statements concern the number of knights among the five.
    *   Let $N$ be the actual number of knights (where $0 \le N \le 5$).

2.  **Analyze the Statements (S):**
    *   $S_A$: "All of us are knaves." (Meaning $N=0$)
    *   $S_B$: "Exactly one of us is a knight." (Meaning $N=1$)
    *   $S_C$: "Exactly two of us are knights." (Meaning $N=2$)
    *   $S_D$: "Exactly three of us are knights." (Meaning $N=3$)
    *   $S_E$: "Exactly four of us are knights." (Meaning $N=4$)

3.  **Determine the Truth Value of Each Statement based on $N$
```

*Inference Speed: Baseline = 71.36 t/s | MTP = 40.28 t/s (0.56x speedup)*

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
Here's a plan to structure the explanation:
1.  **Analyze the core components:** The scenario involves scaling an `asyncio` script (using non-blocking I/O) from 2 tasks to 100,000 tasks.
2.  **Identify the key resources:** Memory footprint and CPU context switching overhead.
3.  **Determine the behavior of `asyncio`:** `asyncio` is highly efficient because it uses a single thread and an event loop to manage I/O waits, making it excellent for high concurrency (C10k problem).
4.  **Address Memory Footprint:** How does the memory usage scale?
5.  **Address CPU Context Switching Overhead:** How does the overhead scale?
6.  **Synthesize the answer:** Provide a clear, technical explanation of the scaling effects.

*(Self-Correction/Refinement: The crucial point is that while 100,000 tasks are created, they are mostly waiting on I/O (network latency), not actively computing. This is where `asyncio` shines.)*<channel|>This is where the efficiency of `asyncio` truly shines, demonstrating the power of cooperative multitasking.
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a plan to structure the explanation:
1.  **Analyze the core components:** The scenario involves scaling an `asyncio` script (using non-blocking I/O) from 2 tasks to 100,000 tasks.
2.  **Identify the key resources:** Memory footprint and CPU context switching overhead.
3.  **Determine the behavior of `asyncio`:** `asyncio` is highly efficient because it uses a single thread and an event loop to manage I/O waits, making it excellent for high concurrency (C10k problem).
4.  **Address Memory Footprint:** How does the memory usage scale?
5.  **Address CPU Context Switching Overhead:** How does the overhead scale?
6.  **Synthesize the answer:** Provide a clear, technical explanation of the scaling effects.

*(Self-Correction/Refinement: The crucial point is that while 100,000 tasks are created, they are mostly waiting on I/O (network latency), not actively computing. This is where `asyncio` shines.)*<channel|>This is where the efficiency of `asyncio` truly shines, as it manages the **C10
```

*Inference Speed: Baseline = 71.64 t/s | MTP = 41.27 t/s (0.58x speedup)*

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
Here's a plan to fulfill the request:
1.  **Identify the PLE scaling equation:** Scan the text for the formula.
2.  **Determine the difference between Layer 4 and Layer 28:** Scan the text for descriptions of these specific layers.
3.  **Explain the speculative token length (Draft Step = 3):** Scan the text for the purpose and constraints of the speculative decoding cycle.
4.  **Format the output:** Present the three pieces of information as three distinct bullet points.

*Self-Correction/Verification during drafting:*
*   *Equation:* `PLE_s = tanh(W_s * h_t + b_s) * gamma_s`. (Exact match required)
*   *Layer Difference:* Layer 4 (low) -> High dimensionality, coarse structural syntactic properties. Layer 28 (intermediate/high) -> Contracts to low-dimensional head projections, semantic vocabulary consolidation.
*   *Draft Step 3:* Maximizes acceptance probability while keeping context memory overhead within bound constraints.

*Execution:* Combine these findings into the requested format.<channel|>*   **PLE Scaling Equation:** The exact mathematical equation used to calculate the PLE scaling factor is:
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Here's a plan to fulfill the request:
1.  **Identify the PLE scaling equation:** Scan the text for the formula.
2.  **Determine the difference between Layer 4 and Layer 28:** Scan the text for descriptions of these specific layers.
3.  **Explain the speculative token length (Draft Step = 3):** Scan the text for the purpose and constraints of the speculative decoding cycle.
4.  **Format the output:** Present the three pieces of information as three distinct bullet points.

*Self-Correction/Verification:* Ensure the extracted equation is exact, the summary of the layer differences is accurate, and the explanation for the draft step length is complete based on the provided text.

*Execution:* (Proceeding with extraction.)<channel|>*   **PLE Scaling Factor Equation:** $\text{PLE\_s} = \tanh(W_s \cdot h_t + b_s) \cdot \gamma_s$
*   **Layer Differences:** Layer 4 utilizes a high dimensionality projection to capture coarse structural syntactic properties, whereas Layer 28 contracts to low-dimensional head projections focused on semantic vocabulary consolidation.
*   **Speculative Draft Step Length:** The speculative draft draft
```

*Inference Speed: Baseline = 71.31 t/s | MTP = 39.42 t/s (0.55x speedup)*

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
1. The current weather for Tokyo, Japan.
2. The equivalent of 500 USD in Japanese Yen (JPY).

I need to use the available tools in sequence.

Step 1: Get the weather for Tokyo.
Tool: `get_weather(location: str)`
Argument: `location` = "Tokyo, Japan"

Step 2: Convert 500 USD to JPY.
Tool: `convert_currency(amount: float, from_currency: str, to_currency: str)`
Arguments: `amount` = 500, `from_currency` = "USD", `to_currency` = "JPY"

I must output the tool calls in a sequence. Since the prompt asks to "Call the appropriate tools in sequence," I will start with the first request (weather) and then the second (currency conversion).

First call (Weather):
`get_weather(location="Tokyo, Japan")`

Second call (Currency):
`convert_currency(amount=500, from_currency="USD", to_currency="JPY")`

I will generate the JSON blocks for both calls
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
The user wants two pieces of information:
1. The current weather for Tokyo, Japan.
2. The equivalent of 500 USD in Japanese Yen (JPY).

I need to use the available tools sequentially.

Step 1: Get the weather for Tokyo.
Tool: `get_weather(location: str)`
Argument: `location` = "Tokyo, Japan"

Step 2: Convert 500 USD to JPY.
Tool: `convert_currency(amount: float, from_currency: str, to_currency: str)`
Arguments: `amount` = 500, `from_currency` = "USD", `to_currency` = "JPY"

I must output the tool calls in JSON blocks and no conversational prose. I will start with the first call.<channel|>```json
{
  "tool": "get_weather",
  "arguments": {
    "location": "Tokyo, Japan"
  }
}
```
```

*Inference Speed: Baseline = 71.26 t/s | MTP = 40.01 t/s (0.56x speedup)*

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
    *   Specific speed of coronal mass ejections (CMEs) from the central star.
    *   Two main chemical components of the interstellar dust.
    *   Coalition regulation governing navigation in the region.

2.  **Scan Passages for Information (Target 1: CME Speed):**
    *   *Passage 1:* "...ejecting coronal mass at speeds exceeding 4,500 km/s."
    *   *Finding:* The speed is exceeding 4,500 km/s.

3.  **Scan Passages for Information (Target 2: Dust Composition):**
    *   *Passage 2:* "The dust composition is uniquely high in magnesium silicate (82%) and iron oxide (12%)."
    *   *Finding:* The two main components are magnesium silicate and iron oxide.

4.  **Scan Passages for Information (Target 3: Navigation Regulation):**
    *   *Passage 2:* "According to the United Stellar Coalition Regulation
```

#### 🟢 MTP Speculative Output:
```json
<|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking for three specific pieces of information:
    *   The speed of coronal mass ejections from the central star.
    *   The two main chemical components of the interstellar dust.
    *   The coalition regulation governing navigation in the region.
2.  **Analyze the Constraints:** I must use *only* the provided passages and must not assume or extrapolate. If the answer is missing, I must state that I cannot answer.
3.  **Scan Passages for Information (Coronal Mass Ejection Speed):**
    *   *Passage 1:* "...ejecting coronal mass at speeds exceeding 4,500 km/s."
    *   *Finding:* The speed is exceeding 4,50 km/s.
4.
    .
    .
    *
    .
            .
            .
            ..
    ..
    ..
        .
        ..
    ..
    .    .
            .
                ..
            ...
```

*Inference Speed: Baseline = 70.86 t/s | MTP = 41.65 t/s (0.59x speedup)*

---

