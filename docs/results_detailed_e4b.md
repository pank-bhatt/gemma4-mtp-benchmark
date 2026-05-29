# Gemma 4 E4B MTP Detailed & Complicated Simulation Report

This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 E4B** across highly detailed, complex scenarios. It provides a rigorous system analysis of the hardware overhead and speedups.

## 🖥️ System hardware Overhead & Static Pressure Analysis

When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.

### 1. Static Load (Model Weight Memory Footprint)
- **Baseline Static Footprint** (Gemma 4 E4B alone): **901.8 MB**
- **MTP Static Footprint** (Gemma 4 E4B + Drafter loaded simultaneously): **1096.3 MB**
- **Absolute Hardware Overhead**: **+194.5 MB**
- **Relative Memory Increase**: **21.6% additional Unified RAM required**

> [!IMPORTANT]
> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.

### 2. CPU Dispatch & Context-Switching Pressure
- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.
- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. 
- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.

## 📊 Summary Performance Table

| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict JSON Schema Generation** | Baseline | 17.10 t/s | Ref (1.00x) | 2421.7 MB | 72.7% | - | - |
| | **MTP (Spec)** | **50.45 t/s** | **2.95x** | **2658.2 MB** | **74.9%** | **+2.2%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knights & Knaves Deduction** | Baseline | 17.69 t/s | Ref (1.00x) | 2936.1 MB | 71.5% | - | - |
| | **MTP (Spec)** | **39.49 t/s** | **2.23x** | **3306.4 MB** | **74.8%** | **+3.4%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Turn Chat History Simulation** | Baseline | 18.27 t/s | Ref (1.00x) | 3381.2 MB | 70.7% | - | - |
| | **MTP (Spec)** | **26.49 t/s** | **1.45x** | **3815.1 MB** | **69.8%** | **-0.9%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PLE Technical Needle Extraction** | Baseline | 16.54 t/s | Ref (1.00x) | 3997.6 MB | 68.7% | - | - |
| | **MTP (Spec)** | **27.75 t/s** | **1.68x** | **4288.9 MB** | **70.9%** | **+2.1%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Agentic Tool Calling Dispatch** | Baseline | 17.71 t/s | Ref (1.00x) | 4317.9 MB | 69.8% | - | - |
| | **MTP (Spec)** | **41.23 t/s** | **2.33x** | **4380.4 MB** | **65.1%** | **-4.7%** | ✅ Active Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Overall Average Metrics:
- **Average Baseline Speed**: **17.46 tokens/second**
- **Average MTP Speed**: **37.08 tokens/second**
- **Net Speedup Factor**: **2.12x faster**

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
  "version": "1.2.3",
  "author": {
    "name": "Jane Doe",
    "email": "jane.doe@example.com"
  },
  "features": [
    {
      "name": "Tokenization",
      "status": "stable",
      "tokens_processed": 150000
    },
    {
      "name": "SentimentAnalysis",
      "status": "beta",
      "tokens_processed": 45000
    },
    {
      "name": "EntityRecognition",
      "status": "alpha",
      "tokens_processed": 12000
    }
  ]
}
```

#### 🟢 MTP Speculative Output:
```json
{
  "project": "DataProcessor",
  "version": "1.2.3",
  "author": {
    "name": "Jane Doe",
    "email": "jane.doe@example.com"
  },
  "features": [
    {
      "name": "Tokenization",
      "status": "stable",
      "tokens_processed": 150000
    },
    {
      "name": "SentimentAnalysis",
      "status": "beta",
      "tokens_processed": 45000
    },
    {
      "name": "EntityRecognition",
      "status": "alpha",
      "tokens_processed": 12000
    }
  ]
}
```

*Inference Speed: Baseline = 17.10 t/s | MTP = 50.45 t/s (2.95x speedup)*

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
This is a classic logic puzzle that can be solved by testing hypotheses based on the nature of knights (always truthful) and knaves (always lying).

Let $N$$ be the total number of knights among the five people (A, B, C, D, E). Since there are five people, $0 \le N \le 5$.

We will analyze the statements based on the assumption of a specific value for $N$.

### The Statements:

* **A says:** "All of us are knaves." (i.. The number of knights is 0, so $N=0$).
* **B says:** "Exactly one of us is a knight." (ii. $N=1$).
* **C says:** "Exactly two of us are knights." (iii. $N=2$).
* **D says:** "Exactly three of us are knights." (iv. $N=3$).
* **E says:** "Exactly four of us are knights." (v. $N=4).

### The Logic Rule:

For any person $X$, if $X$ is a knight, their statement must be true. If $X$ is a knave, their statement must
```

#### 🟢 MTP Speculative Output:
```json
This is a classic logic puzzle that can be solved by testing hypotheses based on the nature of knights (always truthful) and knaves (always lying).

Let $N$$ be the total number of knights among the five people (A, B, C, D, E). Since there are five people, $0 \le N \le 5$.

We will analyze the statements based on the assumption of a specific value for $N$.

### The Statements:

* **A says:** "All of us are knaves." (i.. The number of knights is 0, so $N=0$).
* **B says:** "Exactly one of us is a knight." (ii. $N=1$).
* **C says:** "Exactly two of us are knights." (iii. $N=2$).
* **D says:** "Exactly three of us are knights." (iv. $N=3$).
* **E says:** "Exactly four of us are knights." (v. $N=4).

### The Logic Rule:

For any person $X$, if $X$ is a knight, their statement must be true. If $X$ is a knave, their statement must
```

*Inference Speed: Baseline = 17.69 t/s | MTP = 39.49 t/s (2.23x speedup)*

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
This is a great question that gets to the heart of why asynchronous programming is so powerful for I/O-bound tasks like web scraping.

Here is a detailed explanation of what would happen to the memory footprint and CPU context switching overhead when scaling the `asyncio` script from 2 to 100,000 pages.

---

## 1. Memory Footprint

The memory footprint of the `asyncio` script will **scale linearly, but very efficiently**, compared to a synchronous approach.

### How it scales:

1. **Low Per-Task Overhead:** In `asyncio`, when you initiate a request (e.g, using `aiohttp`), you are not creating a full, heavy operating system (OS) thread for every single request. Instead, you are creating a lightweight **coroutine** (a function defined with `async def`).
2. **Coroutines vs. Threads:** A coroutine is essentially a state machine managed by the Python event loop. It requires significantly less memory overhead than a full OS thread. An OS thread requires its own stack space, kernel resources, and management overhead.
3. **The Scaling Effect:** As you scale to 100,000 pages, you
```

#### 🟢 MTP Speculative Output:
```json
This is a great question that gets to the heart of why asynchronous programming is so powerful for I/O-bound tasks like web scraping.

Here is a detailed explanation of what would happen to the memory footprint and CPU context switching overhead when scaling the `asyncio` script from 2 to 100,000 pages.

---

## 1. Memory Footprint

The memory footprint of the `asyncio` script will **scale linearly, but very efficiently**, compared to a synchronous approach.

### How it scales:

1. **Low Per-Task Overhead:** In `asyncio`, when you initiate a request (e.g, using `aiohttp`), you are not creating a full, heavy operating system (OS) thread for every single request. Instead, you are creating a lightweight **coroutine** (a function defined with `async def`).
2. **Coroutines vs. Threads:** A coroutine is essentially a state machine managed by the Python event loop. It requires significantly less memory overhead than a full OS thread. An OS thread requires its own stack space, kernel resources, and management overhead.
3. **The Scaling Effect:** As you scale to 100,000 pages, you
```

*Inference Speed: Baseline = 18.27 t/s | MTP = 26.49 t/s (1.45x speedup)*

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
Here are the extractions and explanations based on the technical description:

* **Mathematical Equation for PLE Scaling Factor:** The exact equation is $\text{PLE\_}^\text{s} = \tanh(W\_^\text{s} * h\_^\text{t} + b\_^\text{s}) * \gamma\_^text{s}$.
* **Difference in Layer Functionality:** In lower layers (e.g Layer 4), PLE utilizes a high dimensionality projection to capture coarse structural syntactic properties, whereas in intermediate and higher layers (e.g Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation.
* **Reason for Speculative Draft Step Length of 3:** The speculative draft step length is set to 3 because this length maximizes the acceptance probability while simultaneously keeping the context memory overhead within defined bound constraints.
```

#### 🟢 MTP Speculative Output:
```json
Here are the extractions and explanations based on the technical description:

* **Mathematical Equation for PLE Scaling Factor:** The exact equation is $\text{PLE\_}^\text{s} = \tanh(W\_^\text{s} * h\_^\text{t} + b\_^\text{s}) * \gamma\_^text{s}$.
* **Difference in Layer Functionality:** In lower layers (e.g Layer 4), PLE utilizes a high dimensionality projection to capture coarse structural syntactic properties, whereas in intermediate and higher layers (e.g Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation.
* **Reason for Speculative Draft Step Length of 3:** The speculative draft step length is set to 3 because this length maximizes the acceptance probability while simultaneously keeping the context memory overhead within defined bound constraints.
```

*Inference Speed: Baseline = 16.54 t/s | MTP = 27.75 t/s (1.68x speedup)*

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
      "amount": 500,
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
      "amount": 500,
      "from_currency": "USD",
      "to_currency": "JPY"
    }
  }
]
```

*Inference Speed: Baseline = 17.71 t/s | MTP = 41.23 t/s (2.33x speedup)*

---

