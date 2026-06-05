# Automated Dialogue Summarization Engine

## Executive Metadata
* **Target Architecture:** Cross-Attention Hybrid Transformer (BERT-Base Encoder $\rightarrow$ GPT-2 Decoder)
* **Downstream Dataset:** SAMSum Corpus (Messenger-Style Dialogues)
* **Infrastructure Context:** Distributed Accelerators (Google Colab CUDA Environment)

---

## The Business Problem: Information Overload
In modern messaging platforms, group chats serve as a primary engine for user interaction. However, when users disconnect from the platform for even a brief interval, they frequently return to hundreds of unread messages. Crucial details—such as scheduling logistics, action items, or critical updates—are buried under a high-volume stream of conversational tangents and noise. 

### Key Platform Impact Metrics
* **Temporal Friction:** The average user spends **15 to 20 minutes daily** attempting to manually scroll backward through conversational history to re-establish contextual awareness.
* **Psychological Fatigue:** Up to **68% of active users** report explicit platform anxiety regarding hidden or missed notifications within group threads.
* **Retention Churn:** Overwhelmed users frequently utilize the "Mute Thread" feature, which correlates with an immediate drop-off in platform session lengths and overall Daily Active Users (DAU).

### The Proposed Solution
To mitigate this cognitive load, we proposed an **Automated Dialogue Summarization Feature**. By utilizing a hybrid AI architecture trained on real-world messenger-style dialogues (SAMSum), the pipeline condenses chaotic chat histories into structured, readable summaries. This transforms a 200-message scroll into a 3-line summary, allowing users to capture vital context in **5 seconds instead of 5 minutes**.

### Engineering Success Criteria
Model performance is tracked using a combination of qualitative human evaluation and quantitative ROUGE metrics against strict corporate target thresholds:

| Metric | Target Threshold | Functional Objective |
| :--- | :--- | :--- |
| **ROUGE-1** | `> 45.0%` | Ensures key vocabulary and named entities are captured. |
| **ROUGE-2** | `> 22.0%` | Ensures local phrase continuity and local syntax accuracy. |
| **ROUGE-L** | `> 40.0%` | Ensures overall structural fluidity and sentence-level alignment. |

---

## Technical Approach & Implementation Pipeline
The technical pipeline detailed below represents the primary baseline model. While an alternative framework utilizing progressive layer freezing and advanced prompt engineering was developed, infrastructure and credit limits on free-tier computing cut its optimization phase short.

### 1. Data Exploration & Preprocessing
Initial data profiling involved evaluating split distributions, identifying duplicate/missing values, and parsing structural context lengths. Due to the highly erratic capitalization, non-uniform speaker naming conventions, and embedded media tags in real-world chat data, we implemented a custom standardization pipeline:
* **Speaker Variable Normalization:** Raw speaker identities (e.g., "Jane", "John") were algorithmically mapped to static variables (`Speaker 1`, `Speaker 2`). This prevents the encoder from learning individual name tokens as core structural components, forcing the model to focus purely on conversational semantics.
* **Context Window Expansion:** Increased the encoder input ceiling from 256 to **384 tokens** to guarantee that long-form conversations are completely captured without edge truncation.
* **Cross-Entropy Loss Masking:** Explicitly mapped trailing padding tokens to a static value of `-100`. This instructs the PyTorch loss engine to bypass unmasked padding zones, ensuring gradient updates are driven solely by active text tokens.

### 2. Architecture Design
We implemented a custom **Hybrid Encoder-Decoder Model** pairing a `BERT-Base-Uncased` encoder with a `GPT-2` decoder. This configuration was selected after testing and scrapping alternative setups:
1. *BERT-to-T5 Hybrid:* Failed due to inherent architectural limits when the T5 decoder attempted to project and interpret BERT's hidden state embeddings.
2. *BERT-to-BERT Hybrid:* Scrapped because the bidirectional BERT decoder was structurally too weak to handle fluid, autoregressive casual text generation.
3. *BERT-to-GPT-2 Baseline:* Selected as the optimal architecture due to its hardware efficiency, ease of structural alignment via cross-attention layers, and strong generation capabilities.

### 3. Fine-Tuning & Inference Configurations
Training was executed over a high-density, sampled subset of 7,000 conversational entries using accelerated CUDA hardware. Training stability was enforced via the following runtime parameters:

```json
{
  "optimizer_settings": {
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16,
    "early_stopping_patience": 1
  },
  "inference_generation_controls": {
    "num_beams": 4,
    "temperature": 0.1,
    "no_repeat_ngram_size": 3,
    "max_length": 25,
    "early_stopping": true
  }
}

