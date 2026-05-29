# Gemma 4 E2B MTP Detailed & Complicated Simulation Report

This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 E2B** across highly detailed, complex scenarios. It provides a rigorous system analysis of the hardware overhead and speedups.

## 🖥️ System hardware Overhead & Static Pressure Analysis

When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.

### 1. Static Load (Model Weight Memory Footprint)
- **Baseline Static Footprint** (Gemma 4 E2B alone): **734.0 MB**
- **MTP Static Footprint** (Gemma 4 E2B + Drafter loaded simultaneously): **836.2 MB**
- **Absolute Hardware Overhead**: **+102.3 MB**
- **Relative Memory Increase**: **13.9% additional Unified RAM required**

> [!IMPORTANT]
> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.

### 2. CPU Dispatch & Context-Switching Pressure
- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.
- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. 
- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.

## 📊 Summary Performance Table

| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Schema Generation** | Baseline | 24.06 t/s | Ref (1.00x) | 2191.4 MB | 87.0% | - | - |
| | **MTP (Spec)** | **60.32 t/s** | **2.51x** | **2436.4 MB** | **83.6%** | **-3.4%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knights & Knaves Deduction** | Baseline | 26.56 t/s | Ref (1.00x) | 2710.2 MB | 86.8% | - | - |
| | **MTP (Spec)** | **55.98 t/s** | **2.11x** | **3057.5 MB** | **83.8%** | **-3.1%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Turn Chat History Simulation** | Baseline | 28.38 t/s | Ref (1.00x) | 3130.6 MB | 86.9% | - | - |
| | **MTP (Spec)** | **30.18 t/s** | **1.06x** | **3704.2 MB** | **84.4%** | **-2.5%** | ⚠️ Rejected Draft Overhead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PLE Technical Needle Extraction** | Baseline | 25.50 t/s | Ref (1.00x) | 3822.3 MB | 84.7% | - | - |
| | **MTP (Spec)** | **45.53 t/s** | **1.79x** | **4005.6 MB** | **82.2%** | **-2.4%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Agentic Tool Calling Dispatch** | Baseline | 26.41 t/s | Ref (1.00x) | 4033.0 MB | 87.3% | - | - |
| | **MTP (Spec)** | **69.43 t/s** | **2.63x** | **4086.9 MB** | **81.7%** | **-5.6%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Overall Average Metrics:
- **Average Baseline Speed**: **26.18 tokens/second**
- **Average MTP Speed**: **52.29 tokens/second**
- **Net Speedup Factor**: **2.00x faster**

## 📝 Detailed Prompt & Generations Log

### Scenario 1: Strict JSON Schema Generation
**Prompt:**
```
Generate a valid JSON object matching the following schema. The JSON must contain a 'project' key (string), a 'version' key (semver), an 'author' key with 'name' and 'email', a 'features' key which is a list of objects, each having 'name' (string), 'status' (enum: stable, beta, alpha), and 'tokens_processed' (integer). Include exactly 3 features. Do not wrap in markdown tags or include any prose.
```

#### 🔴 Standard Baseline Output:
```json
{
  "project": "DataProcessor",
  "version": "1.2.0",
  "author": {
    "name": "Jane Doe",
    "email": "jane.doe@example.com"
  },
  "features": [
    {
      "name": "Data Ingestion",
      "status": "stable",
      "tokens_processed": 150000
    },
    {
      "name": "Real-time Analysis",
      "status": "beta",
      "tokens_processed": 50000
    },
    {
      "name": "API Integration",
      "status": "alpha",
      "tokens_processed": 10000
    }
  ]
}
```

#### 🟢 MTP Speculative Output:
```json
{
  "project": "DataProcessor",
  "version": "1.2.0",
  "author": {
    "name": "Jane Doe",
    "email": "jane.doe@example.com"
  },
  "features": [
    {
      "name": "Data Ingestion",
      "status": "stable",
      "tokens_processed": 150000
    },
    {
      "name": "Real-time Analysis",
      "status": "beta",
      "tokens_processed": 50000
    },
    {
      "name": "API Integration",
      "status": "alpha",
      "tokens_processed": 10000
    }
  ]
}
```

*Inference Speed: Baseline = 24.06 t/s | MTP = 60.32 t/s (2.51x speedup)*

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
Let $N$ be the total number of people, so $N=5$.
Let $K$ be the actual number of knights in the room, so $K \in \{0, 1, 2, 3, 4, 5\}$.
Let $S_X$ be the statement made by person $X$.
If person $X$ is a knight, then $S_X$ is true.
If person $X$ is a knave, then $S_X$ is false.

The statements are:
A says ($S_A$): "All of us are knaves." (i.e., $K=0$)
B says ($S_B$): "Exactly one of us is a knight." (i.e., $K=1$)
C says ($S_C$): "Exactly two of us are knights." (i.e., $K=2$)
D says ($S_D$): "Exactly three of us are knights." (i.e., $K=3$)
E says ($S_E$): "Exactly four of us are knights." (i.e., $K=4$)

We will test each
```

#### 🟢 MTP Speculative Output:
```json
Let $N$ be the total number of people, so $N=5$.
Let $K$ be the actual number of knights in the room, so $K \in \{0, 1, 2, 3, 4, 5\}$.
Let $S_X$ be the statement made by person $X$.
If person $X$ is a knight, then $S_X$ is true.
If person $X$ is a knave, then $S_X$ is false.

The statements are:
A says ($S_A$): "All of us are knaves." (i.e., $K=0$)
B says ($S_B$): "Exactly one of us is a knight." (i.e., $K=1$)
C says ($S_C$): "Exactly two of us are knights." (i.e., $K=2$)
D says ($S_D$): "Exactly three of us are knights." (i.e., $K=3$)
E says ($S_E$): "Exactly four of us are knights." (i.e., $K=4$)

We will test each
```

*Inference Speed: Baseline = 26.56 t/s | MTP = 55.98 t/s (2.11x speedup)*

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
This is a great question that gets to the heart of why asynchronous programming is beneficial for I/O-bound tasks.

Here is a breakdown of what would happen to the **memory footprint** and **CPU context switching overhead** when scaling an `asyncio` script from 2 pages to 100,000 pages.

---

## 1. Memory Footprint

The memory footprint of the `asyncio` script will generally be **significantly lower and more stable** than a purely synchronous script, especially as the number of concurrent operations increases.

### Synchronous Script Memory Behavior:

* **High Memory Usage:** A synchronous script that fetches 100,000 pages sequentially (or even concurrently using threads) would likely require memory to hold the state for *all* pending operations simultaneously. If you used threads, each thread would consume its own stack space, leading to a large, potentially unbounded memory footprint.
* **Blocking:** While waiting for a network response for one page, the entire process (or thread) is blocked, but the memory allocated for that blocked operation remains active.

### Asynchronous (`asyncio`) Script Memory Behavior:

* **Low, Predictable Memory Usage:** `asyncio` operates
```

#### 🟢 MTP Speculative Output:
```json
This is a great question that gets to the heart of why asynchronous programming is beneficial for I/O-bound tasks.

Here is a breakdown of what would happen to the **memory footprint** and **CPU context switching overhead** when scaling an `asyncio` script from 2 pages to 100,000 pages.

---

## 1. Memory Footprint

The memory footprint of the `asyncio` script will generally be **significantly lower and more stable** than a purely synchronous script, especially as the number of concurrent operations increases.

### Synchronous Script Memory Behavior:

* **High Memory Usage:** A synchronous script that fetches 100,000 pages sequentially (or even concurrently using threads) would likely require memory to hold the state for *all* pending operations simultaneously. If you used threads, each thread would consume its own stack space, leading to a large, potentially unbounded memory footprint.
* **Blocking:** While waiting for a network response for one page, the entire process (or thread) is blocked, but the memory allocated for that blocked operation remains active.

### Asynchronous (`asyncio`) Script Memory Behavior:

* **Low, Predictable Memory Usage:** `asyncio` operates
```

*Inference Speed: Baseline = 28.38 t/s | MTP = 30.18 t/s (1.06x speedup)*

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
Here are the extracted points based on the technical description:

*   **PLE Scaling Equation:** The exact mathematical equation used to calculate the PLE scaling factor is: $\text{PLE\_s} = \tanh(W_s \cdot h_t + b_s) \cdot \gamma_s$.
*   **Layer Differences:** In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to capture coarse structural syntactic properties, whereas in intermediate and higher layers (e.g., Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation.
*   **Speculative Token Length:** The speculative token length draft step is set to 3 because this length maximizes the acceptance probability while keeping the context memory overhead within bound constraints.
```

#### 🟢 MTP Speculative Output:
```json
Here are the extracted points based on the technical description:

*   **PLE Scaling Equation:** The exact mathematical equation used to calculate the PLE scaling factor is: $\text{PLE\_s} = \tanh(W_s \cdot h_t + b_s) \cdot \gamma_s$.
*   **Layer Differences:** In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to capture coarse structural syntactic properties, whereas in intermediate and higher layers (e.g., Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation.
*   **Speculative Token Length:** The speculative token length draft step is set to 3 because this length maximizes the acceptance probability while keeping the context memory overhead within bound constraints.
```

*Inference Speed: Baseline = 25.50 t/s | MTP = 45.53 t/s (1.79x speedup)*

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
      "amount": 500.0,
      "from_currency": "USD",
      "to_currency": "JPY"
    }
  }
]
```

#### 🟢 MTP Speculative Output:
```json
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
      "amount": 500.0,
      "from_currency": "USD",
      "to_currency": "JPY"
    }
  }
]
```

*Inference Speed: Baseline = 26.41 t/s | MTP = 69.43 t/s (2.63x speedup)*

---

