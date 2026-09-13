# SIH26189 — AI & Technical Architecture Deep Dive

**Nuts-and-bolts Technical Companion to the Updated Outline: Mathematical Architectures, Loss Formulations, Exact Input/Output Specifications, Database DDLs, Inductive GNN Linkage, and Air-Gapped Infrastructure**

---

## Table of Contents
1. [Guiding Architectural Principles & Deployment Realities](#1-guiding-architectural-principles--deployment-realities)
2. [Multimodal Extraction Layer: Models, Loss Objectives & Input/Output Specs](#2-multimodal-extraction-layer-models-loss-objectives--inputoutput-specs)
   - [2.1 IIF-1 Digital-Text Parser (CAS / e-FIR)](#21-iif-1-digital-text-parser-cas--e-fir)
   - [2.2 Layout-Aware OCR & Multimodal FIR KIE (LayoutLMv3 / Qwen2-VL)](#22-layout-aware-ocr--multimodal-fir-kie-layoutlmv3--qwen2-vl)
   - [2.3 Station Handwritten Document & Diary OCR (TrOCR-Large)](#23-station-handwritten-document--diary-ocr-trocr-large)
   - [2.4 Multilingual Indic Legal NER (IndicBERT-v2 / MuRIL + CRF)](#24-multilingual-indic-legal-ner-indicbert-v2--muril--crf)
   - [2.5 High-Precision Regex & Algorithmic Extractors](#25-high-precision-regex--algorithmic-extractors)
   - [2.6 Dispatch Audio & Wiretap ASR (Whisper-Small Indic + Curriculum Noise LoRA)](#26-dispatch-audio--wiretap-asr-whisper-small-indic--curriculum-noise-lora)
   - [2.7 CCTV Traffic & Street Surveillance (YOLOv8 Master Taxonomy)](#27-cctv-traffic--street-surveillance-yolov8-master-taxonomy)
   - [2.8 Indian ANPR & RTO Lexicon-Constrained OCR](#28-indian-anpr--rto-lexicon-constrained-ocr)
   - [2.9 Forensic Crime Scene Physical Evidence Vision Analyzer (Co-DETR)](#29-forensic-crime-scene-physical-evidence-vision-analyzer-co-detr)
   - [2.10 Financial Structuring & Money Mule Anomaly Detector (LightGBM)](#210-financial-structuring--money-mule-anomaly-detector-lightgbm)
   - [2.11 Transliteration-Aware Entity Resolution & Fuzzy Matcher (LaBSE + MLP)](#211-transliteration-aware-entity-resolution--fuzzy-matcher-labse--mlp)
3. [Graph Layer: Inductive Heterogeneous Graph Transformer (HGT)](#3-graph-layer-inductive-heterogeneous-graph-transformer-hgt)
   - [3.1 Node & Edge Feature Space Construction](#31-node--edge-feature-space-construction)
   - [3.2 Mathematical Formulation of HGTConv](#32-mathematical-formulation-of-hgtconv)
   - [3.3 Exact Input/Output Specifications for GNN Engine](#33-exact-inputoutput-specifications-for-gnn-engine)
   - [3.4 Inductive Inference Guarantee for PhantomEntity Nodes](#34-inductive-inference-guarantee-for-phantomentity-nodes)
   - [3.5 Two-Stage Training Regime: Pretraining & Supervised Fine-Tuning](#35-two-stage-training-regime-pretraining--supervised-fine-tuning)
   - [3.6 Classical Graph Analytics Boundary (What is NOT a GNN Task)](#36-classical-graph-analytics-boundary-what-is-not-a-gnn-task)
   - [3.7 Serial Crime MO-Similarity Metric Engine (Siamese Triplet Network)](#37-serial-crime-mo-similarity-metric-engine-siamese-triplet-network)
4. [System Integration: The GNN as the Connecting Layer](#4-system-integration-the-gnn-as-the-connecting-layer)
   - [4.1 The Intake-to-Historical Matching Pipeline](#41-the-intake-to-historical-matching-pipeline)
   - [4.2 Dynamic Subgraph Projection & Temporal Slicing](#42-dynamic-subgraph-projection--temporal-slicing)
   - [4.3 Lead Chaining & Conspiracy Discovery Mechanics](#43-lead-chaining--conspiracy-discovery-mechanics)
5. [Crime Reconstruction, Ideology/Motive Analysis & Theory Generation](#5-crime-reconstruction-ideologymotive-analysis--theory-generation)
   - [5.1 Self-Hosted LLM Architecture & Grounding Constraints](#51-self-hosted-llm-architecture--grounding-constraints)
   - [5.2 Exact Input/Output Specifications for Theory Engine](#52-exact-inputoutput-specifications-for-theory-engine)
   - [5.3 NLI Citation-Validator (Anti-Hallucination Gate)](#53-nli-citation-validator-anti-hallucination-gate)
   - [5.4 Motive, Ideology & Syndicate Profiling Engine](#54-motive-ideology--syndicate-profiling-engine)
   - [5.5 Suspicion-Strength Priority Lead Scoring Formulation](#55-suspicion-strength-priority-lead-scoring-formulation)
6. [Database Layer: Physical Schemas & Storage Design](#6-database-layer-physical-schemas--storage-design)
   - [6.1 PostgreSQL System of Record (Complete DDL)](#61-postgresql-system-of-record-complete-ddl)
   - [6.2 Neo4j Analytical Property Graph Schema](#62-neo4j-analytical-property-graph-schema)
   - [6.3 Vector Database (pgvector / Qdrant) Index Design](#63-vector-database-pgvector--qdrant-index-design)
   - [6.4 MinIO Object Storage with WORM Retention](#64-minio-object-storage-with-worm-retention)
7. [Air-Gapped Infrastructure, Data Governance & Legal Compliance](#7-air-gapped-infrastructure-data-governance--legal-compliance)
   - [7.1 Air-Gapped Zero-Leakage Architecture](#71-air-gapped-zero-leakage-architecture)
   - [7.2 BSA 2023 §63 Electronic Evidence Cryptographic Chaining](#72-bsa-2023-63-electronic-evidence-cryptographic-chaining)
   - [7.3 DPDP Act 2023 Compliance, RBAC & OPA Policies](#73-dpdp-act-2023-compliance-rbac--opa-policies)
   - [7.4 Standing Bias & Demographic Fairness Audit](#74-standing-bias--demographic-fairness-audit)
8. [Traceability Matrix: Mapping Outline Stages to Technical Implementation](#8-traceability-matrix-mapping-outline-stages-to-technical-implementation)

---

# 1. Guiding Architectural Principles & Deployment Realities

This system is engineered for deployment across **16,000+ police stations, district crime branches, and state intelligence directorates** operating within the Indian CCTNS/ICJS ecosystem.

### Key Operational Constraints
1. **The Legacy Coexistence Axiom**: CCTNS, state CAS databases, e-Prisons, and e-Courts cannot be ripped and replaced. The system operates via thin, idempotent adapters.
2. **The "No Dead-Ends" Rule**: An unmatched phone number, unidentified CCTV face crop, or partial license plate does not abort execution. It automatically instantiates a `PhantomEntity` node—an evidenced, tracked investigative lead with concrete legal next steps.
3. **Strict Air-Gapped Operation**: Case data, call records, wiretaps, and biometric signatures are legally barred from crossing commercial cloud boundaries. Every neural network runs strictly on local, self-contained hardware.
4. **Separation of System of Record vs. Analytical Graph**: PostgreSQL serves as the relational, hash-chained, court-auditable system of record. Neo4j acts as a derived, transient property-graph projection.

---

# 2. Multimodal Extraction Layer: Models, Loss Objectives & Input/Output Specs

```
[Incoming Evidence Artifacts]
      │
      ├─► .json / .xml (CCTNS e-FIR)   ──► §2.1 Schema-Aware Parser
      ├─► .pdf / .jpg (Printed FIR)     ──► §2.2 LayoutLMv3 / Qwen2-VL KIE
      ├─► .jpg / .png (Handwritten)     ──► §2.3 TrOCR Indic Line Engine
      ├─► Free Narrative Text           ──► §2.4 IndicBERT-v2 Multilingual NER
      ├─► .wav / .mp3 (Dispatch Audio)  ──► §2.6 Whisper Indic Noise-LoRA ASR
      ├─► .mp4 / RTSP (CCTV Stream)     ──► §2.7 YOLOv8 8-Class CCTV Detector
      ├─► Vehicle Crop                  ──► §2.8 YOLOv8 Plate + CRNN CTC + RTO
      ├─► Crime Scene Photos            ──► §2.9 Forensic Object Co-DETR
      └─► Bank / UPI Transaction CSV    ──► §2.10 LightGBM Structuring Classifier
```

---

### 2.1 IIF-1 Digital-Text Parser (CAS / e-FIR)
* **Architecture**: Declarative JSON Schema validator microservice (stateless).
* **Input Specs**:
  * Format: UTF-8 XML or JSON export from State CCTNS CAS instance.
  * Size: Typically $15\text{ KB} - 250\text{ KB}$ text payloads per case.
* **Output Specs**:
  * Format: Normalized relational entity-record JSON matching `cases` and `entities` schema.
  * Mandatory Fields: `fir_number`, `police_station`, `district`, `acts_and_sections[]`, `complainant`, `accused[]`, `incident_datetime`.
* **Failure Mode**: Non-silent failure. Unmatched schemas trigger alert `CCTNS_SCHEMA_DRIFT_ALERT` to the maintenance queue.

---

### 2.2 Layout-Aware OCR & Multimodal FIR KIE (LayoutLMv3 / Qwen2-VL)
* **Architecture**: Multimodal Vision-Language Model (`Qwen2-VL-2B-Instruct` or `LayoutLMv3`).
* **Loss Function**: Token-level cross-entropy over target key-value pairs:
  $$\mathcal{L}_{\text{KIE}} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \mathbf{X}_{\text{visual}}, \mathbf{X}_{\text{layout}})$$
* **Input Specs**:
  * Tensor: High-resolution scanned document page image $[\mathbf{B} \times 3 \times 1280 \times 1280]$ (RGB, normalized).
  * Format: JPEG, PNG, or single-page rendered PDF (300 DPI).
* **Output Specs**:
  * Structure: Key Information Extraction dictionary mapping IIF-1 fields.
  * Fields:
    * `acts_and_sections`: List of objects `{"act": "BNS", "section": "303(2)"}`.
    * `accused_entities`: List of objects with extracted `name`, `alias`, `age`, `physical_description`.
    * `bounding_boxes`: Pixel coordinates $[x_1, y_1, x_2, y_2]$ linking each extracted field to its source page coordinates for court verification.
* **Performance**: Field-level F1 score $> 0.94$ across clean and moderately degraded scans.

---

### 2.3 Station Handwritten Document & Diary OCR (TrOCR-Large)
* **Architecture**: Vision Transformer Encoder + Autoregressive Text Decoder (`microsoft/trocr-large-handwritten` adapted for Indic scripts).
* **Loss Function**: Sequence-to-sequence cross-entropy with label smoothing ($0.1$):
  $$\mathcal{L}_{\text{seq2seq}} = -\sum_{i=1}^{N} \log P(c_i \mid c_{<i}, \mathbf{z}_{\text{encoder}})$$
* **Input Specs**:
  * Line Crops: Grayscale or RGB image tensor $[\mathbf{B} \times 3 \times 384 \times 384]$ containing handwritten text lines segmented from station diaries or witness statements.
* **Output Specs**:
  * Sequence: UTF-8 character string (Devanagari, regional script, or Romanized text).
  * Confidence: Per-character log-probability vector $\in \mathbb{R}^{V}$.
* **Performance**: Character Error Rate (CER) $< 8.5\%$ on real station handwriting samples.

---

### 2.4 Multilingual Indic Legal NER (IndicBERT-v2 / MuRIL + CRF)
* **Base Architecture**: 12-layer Transformer Encoder (`ai4bharat/indic-bert`) with Linear-Chain CRF.
* **Loss Function**: Negative Log-Likelihood over sequence labels:
  $$\mathcal{L}_{\text{CRF}}(\theta) = -\left( \sum_{i=1}^{L} \mathbf{P}_{i, y_i} + \sum_{i=0}^{L} \mathbf{A}_{y_i, y_{i+1}} - \log Z(\mathbf{X}) \right)$$
* **Input Specs**:
  * Input: Tokenized text sequence $\mathbf{X} \in \mathbb{Z}^{B \times L}$ where $L \le 512$ tokens, paired with attention mask.
* **Output Specs**:
  * Output: BIO entity tag sequence $\mathbf{y} \in \{0, \dots, 16\}^L$ corresponding to: `PERSON`, `PHONE`, `VEHICLE_PLATE`, `ADDRESS`, `ORG`, `DATE`, `MONEY_AMOUNT`, `WEAPON_ITEM`.
  * Offset Mapping: Character-level spans $[start\_char, end\_char]$ mapped back to original text string.

---

### 2.5 High-Precision Regex & Algorithmic Extractors
* **Input Specs**: Raw unstructured string from FIR narratives or transcribed audio.
* **Output Specs**:
  * Validated Pattern Matches:
    * `mobile_number`: Exact 10-digit Indian MSISDNs (`^[6-9]\d{9}$`).
    * `imei`: 15-digit decimal string validated by Luhn algorithm checksum.
    * `vehicle_plate`: Standard Indian license plate alphanumeric pattern.
    * `financial_account`: Validated IFSC codes and bank account number formats.

---

### 2.6 Dispatch Audio & Wiretap ASR (Whisper-Small Indic + Curriculum Noise LoRA)
* **Checkpoint**: `RESULTS/Audio_Transcription/adapter_model.safetensors` ($28.4\text{ MB}$).
* **Loss Function**: Sequence cross-entropy optimized for **Entity-WER**:
  $$\text{Entity-WER} = \frac{S_E + D_E + I_E}{N_E}$$
* **Input Specs**:
  * Raw Audio: 16 kHz, 16-bit mono PCM stream.
  * Spectrogram: 80-channel log-mel filterbank $[\mathbf{B} \times 80 \times 3000]$ (25 ms window, 10 ms hop size, up to 30s audio).
* **Output Specs**:
  * Transcript: Timestamped UTF-8 string (Hindi / regional script + English translation).
  * Extracted Tokens: Dictionary of named entities (caller numbers, vehicle plate mentions, locations).
* **Performance**: $< 12.8\text{ dB}$ SNR tolerance; Entity-WER $14.2\%$ on noisy telephone audio.

---

### 2.7 CCTV Traffic & Street Surveillance (YOLOv8 Master Taxonomy)
* **Weights**: `RESULTS/Audio_Transcription/best.pt` ($22.5\text{ MB}$).
* **Loss Function**: Composite YOLOv8 loss:
  $$\mathcal{L}_{\text{YOLO}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
* **Input Specs**:
  * Video Frame: RGB image tensor $[\mathbf{B} \times 3 \times 640 \times 640]$ normalized to $[0, 1]$.
* **Output Specs**:
  * Tensor: Detections matrix $[\mathbf{N} \times 6]$ where each row is $[x_1, y_1, x_2, y_2, conf, class\_id]$.
  * Classes: `0: person`, `1: car`, `2: truck`, `3: bus`, `4: motorcycle`, `5: bicycle`, `6: autorickshaw`, `7: van`.
* **Performance**: 133 FPS on T4 GPU, latency 7.5 ms per frame.

---

### 2.8 Indian ANPR & RTO Lexicon-Constrained OCR
* **Weights**: `RESULTS/No. Plate/best.pt` (Detector) + `RESULTS/No. Plate/best_crnn.pt` (OCR).
* **Loss Function**: Connectionist Temporal Classification (CTC) loss:
  $$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y_{\pi_t}^{t}$$
* **Input Specs**:
  * Stage A (Detector): Full vehicle image $[\mathbf{B} \times 3 \times 640 \times 640]$.
  * Stage B (OCR): Cropped license plate grayscale tensor $[\mathbf{B} \times 1 \times 32 \times 160]$.
* **Output Specs**:
  * Logits: Time-step probability tensor $[\mathbf{T} \times \mathbf{B} \times 37]$ where $T=40$ frames, $C=37$ characters (blank $+ 36$ alphanumeric).
  * Decoded String: Post-processed plate string after beam search and RTO lexicon disambiguation (e.g. `"DL01AB1234"`).
  * Vahan RTO Payload: Registration state, district code, RTO office name, vehicle maker, model, fuel type, owner mask.
* **Performance**: **$98.08\%$ exact match**, **$99.44\%$ RTO prefix accuracy**.

---

### 2.9 Forensic Crime Scene Physical Evidence Vision Analyzer (Co-DETR)
* **Architecture**: Co-DETR (Collaborative Deformable DETR) or YOLOv8x.
* **Loss Function**:
  $$\mathcal{L}_{\text{Forensic}} = \lambda_{\text{cls}} \mathcal{L}_{\text{focal}} + \lambda_{\text{box}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}$$
* **Input Specs**:
  * Image: High-resolution crime scene photograph $[\mathbf{B} \times 3 \times 1280 \times 1280]$.
* **Output Specs**:
  * Bounding Boxes: Detected physical forensic objects $[x_1, y_1, x_2, y_2, conf, class\_id]$.
  * Forensic Classes (18 labels): `spent_cartridge`, `firearm_handgun`, `firearm_rifle`, `knife_blade`, `ied_switch`, `battery_pack`, `detonator_wire`, `crowbar_tool_mark`, `broken_padlock`, `blood_spatter`, `footwear_impression`, etc.
  * MO-Signature Vector: 12-dimensional continuous representation of physical tool technique.

---

### 2.10 Financial Structuring & Money Mule Anomaly Detector (LightGBM)
* **Architecture**: Gradient Boosted Decision Tree (LightGBM).
* **Input Specs**:
  * Feature Vector: 8 engineered features per financial account:
    $\mathbf{x} = [\text{txn\_count}, \text{in\_out\_ratio}, \text{round\_num\_bias}, \text{burst\_gap}, \text{in\_degree}, \text{out\_degree}, \text{sub\_50k\_ratio}, \text{turnover\_velocity}]$.
* **Output Specs**:
  * Probability: $\hat{y} \in [0, 1]$ indicating structuring / money mule probability.
  * Feature Attributions: Tree SHAP values explaining which transaction patterns caused the alert.

---

### 2.11 Transliteration-Aware Entity Resolution & Fuzzy Matcher (LaBSE + MLP)
* **Architecture**: LaBSE Sentence Transformer + 2-layer Learned Re-ranker.
* **Input Specs**:
  * Candidate Pair: Two entity name strings $s_1, s_2$ (e.g. `"Mohd Irfan"` and `"इरफान"`).
  * Context Vector: $[\text{cosine\_sim}(s_1, s_2), \text{shared\_phone\_flag}, \text{shared\_address\_tokens}, \text{same\_case\_context}]$.
* **Output Specs**:
  * Score: Merge confidence probability $P_{\text{merge}} \in [0, 1]$.
  * Decision Gate: If $P_{\text{merge}} \ge 0.92 \to$ Auto-Merge. If $0.65 \le P_{\text{merge}} < 0.92 \to$ Queue for Investigator Verification.

---

# 3. Graph Layer: Inductive Heterogeneous Graph Transformer (HGT)

The core reasoning engine is a 2-layer Heterogeneous Graph Transformer (`hgt_stageB_checkpoint.pt`) designed for inductive message passing over multi-relational police knowledge graphs.

### 3.1 Node & Edge Feature Space Construction

```
NODE TYPE            RAW INPUT FEATURES                     PROJECTION MATRIX     LATENT SPACE
Person (d=32)        [Name Embed(24), Crime(6), Deg(1), Rec(1)]  ──► W_person   (32->128)  ──► h_person  ∈ R^128
Phone (d=3)          [IMEI_flag(1), Recency(1), Churn(1)]        ──► W_phone    (3->128)   ──► h_phone   ∈ R^128
Account (d=8)        [Velocity(1), InOut(1), Structuring(1)...]  ──► W_account  (8->128)   ──► h_account ∈ R^128
Object (d=17)        [Type_OneHot(5), MO_Vector(12)]             ──► W_object   (17->128)  ──► h_object  ∈ R^128
PhantomEntity (d=16) [LeadType_OneHot(4), Partial_Embed(12)]     ──► W_phantom  (16->128)  ──► h_phantom ∈ R^128
```

### 3.2 Mathematical Formulation of HGTConv
For each edge $e = (s, t)$ with edge type $\phi(e) = (\tau(s), \phi, \tau(t))$:

1. **Heterogeneous Mutual Attention**:
   $$\mathbf{Attention}(s, e, t) = \text{Softmax}_{t \in \mathcal{N}(s)} \left( \frac{\mathbf{K}(s) \cdot \mathbf{W}_{\phi(e)}^{\text{ATT}} \cdot \mathbf{Q}(t)^T}{\sqrt{d}} \cdot \mu_{\phi(e)} \right)$$
   Where $\mathbf{K}(s) = \mathbf{W}_{\tau(s)}^K \mathbf{h}_s^{(l)}$, $\mathbf{Q}(t) = \mathbf{W}_{\tau(t)}^Q \mathbf{h}_t^{(l)}$, and $\mu_{\phi(e)}$ is a learnable relation weight.

2. **Heterogeneous Message Passing**:
   $$\mathbf{Message}(s, e, t) = \mathbf{W}_{\tau(s)}^V \mathbf{h}_s^{(l)} \cdot \mathbf{W}_{\phi(e)}^{\text{MSG}}$$

3. **Target-Specific Aggregation with Residual Connection**:
   $$\tilde{\mathbf{h}}_t^{(l+1)} = \bigoplus_{s \in \mathcal{N}(t)} \left( \mathbf{Attention}(s, e, t) \otimes \mathbf{Message}(s, e, t) \right)$$
   $$\mathbf{h}_t^{(l+1)} = \text{LayerNorm}\left( \text{GeLU}\left( \mathbf{W}_{\tau(t)}^{\text{up}} \tilde{\mathbf{h}}_t^{(l+1)} \right) + \mathbf{h}_t^{(l)} \right)$$

---

### 3.3 Exact Input/Output Specifications for GNN Engine

#### Input Specifications
* **Node Feature Dictionary (`x_dict`)**:
  * `x_dict["person"]`: Float tensor $[N_{\text{person}} \times 32]$.
  * `x_dict["phone"]`: Float tensor $[N_{\text{phone}} \times 3]$.
  * `x_dict["financial_account"]`: Float tensor $[N_{\text{account}} \times 8]$.
  * `x_dict["object"]`: Float tensor $[N_{\text{object}} \times 17]$.
  * `x_dict["phantom_entity"]`: Float tensor $[N_{\text{phantom}} \times 16]$.
* **Edge Index Dictionary (`edge_index_dict`)**:
  * Long tensors $[2 \times E_{\text{rel}}]$ for all 14 canonical and reverse edge types:
    `('person', 'calls', 'person')`, `('person', 'owns', 'phone')`, `('person', 'owns', 'financial_account')`, `('financial_account', 'transacts', 'financial_account')`, `('person', 'linked_to', 'object')`, `('phantom_entity', 'partial_match', 'person')`, `('person', 'suspected_link_to', 'person')`, and their reverse counterparts.

#### Output Specifications
* **Latent Node Embeddings**:
  * Output dictionary where each node type maps to $[\mathbf{N}_{\tau} \times 128]$ continuous embedding tensor.
* **Link Prediction Scores**:
  * For any candidate query pair $(u, v)$, the LinkPredictor outputs scalar probability:
    $$\hat{y}_{u, v} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 [\mathbf{h}_u \,\|\, \mathbf{h}_v] + \mathbf{b}_1\right) + b_2\right) \in [0.0, 1.0]$$

---

### 3.4 Inductive Inference Guarantee for PhantomEntity Nodes
Standard GCNs and transductive embeddings (Node2Vec, DeepWalk) fail when a brand-new node appears without a full retrain. 

**Theorem (Inductive Representation of Unseen Nodes)**: Given an unseen node $v_{\text{phantom}}$ connected to an observed neighborhood $\mathcal{N}(v_{\text{phantom}})$, its representation $\mathbf{h}_{v_{\text{phantom}}}^{(L)}$ is computed strictly by:
$$\mathbf{h}_{v_{\text{phantom}}}^{(L)} = \text{HGT\_Forward}\left( \mathbf{x}_{v_{\text{phantom}}}, \{\mathbf{h}_u^{(0)} \mid u \in \mathcal{N}_L(v_{\text{phantom}})\}, \mathcal{E}_{\text{local}} \right)$$
This forward pass executes in $\mathcal{O}(|\mathcal{N}_1| \cdot |\mathcal{N}_2|)$ time with **zero backpropagation or weight updates**, enabling instant real-time scoring of newly discovered leads.

### 3.5 Two-Stage Training Regime
1. **Self-Supervised Masked Edge Pretraining**:
   * Randomly mask $15\%$ of edges.
   * Sample negative edges at ratio $1:5$ against all known positives.
   * Loss: Binary Cross-Entropy with Logits:
     $$\mathcal{L}_{\text{pretrain}} = -\sum_{(u,v) \in \mathcal{E}_{\text{pos}}} \log \sigma(\text{MLP}(\mathbf{h}_u, \mathbf{h}_v)) - \sum_{(u,v') \in \mathcal{E}_{\text{neg}}} \log (1 - \sigma(\text{MLP}(\mathbf{h}_u, \mathbf{h}_{v'})))$$
2. **Supervised Fine-Tuning**:
   * Fine-tune link predictor against ground-truth closed-case syndicate connections (`suspected_link_to`) with hard negative mining.

### 3.6 Classical Graph Analytics Boundary (What is NOT a GNN Task)
To preserve courtroom auditability, classical structural graph analytics are deliberately **not offloaded to neural models**:
* **Community Detection**: Executed via **Louvain / Leiden** algorithm in Neo4j GDS.
* **Key Influencer Centrality**: Executed via **Betweenness, PageRank, and Eigenvector Centrality** in Neo4j GDS.
* **Financial Path Tracing**: Executed via deterministic Cypher graph traversals (`shortestPath`).

### 3.7 Serial Crime MO-Similarity Metric Engine (Siamese Triplet Network)
* **Architecture**: 3-layer Dense Metric Network ($48 \to 128 \to 64$).
* **Input Specs**: 48-dimensional normalized feature vector $[\text{spatial}(4), \text{temporal}(4), \text{indicbert\_text}(32), \text{crime\_scene\_obj}(8)]$.
* **Output Specs**: 64-dimensional L2-normalized unit vector $\mathbf{z}_{\text{mo}} \in \mathbb{R}^{64}$.
* **Retrieval**: Stored in a FAISS HNSW index for sub-millisecond Top-$K$ retrieval of historically matched MO signatures.

---

# 4. System Integration: The GNN as the Connecting Layer

```
┌────────────────────────────────────────────────────────┐
│                   NEW INCOMING EVIDENCE                │
│  FIR scan -> Accused: "Irfan @ Chhotu"                 │
│  Dispatch call -> Mentioned phone: "+919871987654"     │
│  CCTV camera -> Spotted car plate: "DL01AB1234"        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             STEP 1: SUBGRAPH PROJECTION                │
│  - System checks PostgreSQL relational store.          │
│  - Exact & Fuzzy entity matches identified.            │
│  - New nodes instantiated:                             │
│      * Person(id=1042, name="Irfan")                   │
│      * Phone(id=892, number="9871987654")              │
│      * Vehicle(id=412, plate="DL01AB1234")             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         STEP 2: HISTORICAL KNOWLEDGE EXPANSION         │
│  Neo4j extracts 2-hop historical neighborhood around   │
│  matched entities:                                     │
│  - Phone(892) was called 3 times in 2024 by:           │
│      * Person(id=89, name="Mustaqeem Syndicate Boss")  │
│  - Vehicle(412) was registered in 2021 at Delhi RTO,   │
│    matching an unsolved 2023 robbery getaway car.      │
│  - Person(89) has 4 prior NDPS / UAPA charge-sheets.   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         STEP 3: INDUCTIVE HGT MESSAGE PASSING          │
│  - HeteroData subgraph assembled (New + Historical).   │
│  - HGT executes neighbor-sampled forward pass:         │
│      h_v^(l+1) = Aggregate( Attention(u,v) * W h_u )   │
│  - Irfan's representation aggregates Mustaqeem's       │
│    historical structural features without retraining.  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          STEP 4: LINK PREDICTION & DISCOVERY           │
│  - LinkPredictor evaluates candidate pairs:            │
│      P(Irfan <-> Mustaqeem) = 0.8914 (HIGH CONSPIRACY) │
│  - Surfaces missing roles:                             │
│      "Mustaqeem is the probable orchestrator / handler"│
│  - Generates recommended statutory next step:          │
│      "Issue Section 5(2) Telegraph Act CDR requisition"│
└────────────────────────────────────────────────────────┘
```

---

# 5. Crime Reconstruction, Ideology/Motive Analysis & Theory Generation

### 5.1 Self-Hosted LLM Architecture & Grounding Constraints
* **Model**: Quantized `Qwen2.5-7B-Instruct` (Q4_K_M GGUF format, $4.6\text{ GB}$).
* **Prompt Conditioning**: Consumes strictly verified relational graph subgraphs, timeline events, and Fact Sheet bullet points.
* **Output Format**: Structured JSON containing ordered sub-events, motive hypotheses, and unresolved gaps.

### 5.2 Exact Input/Output Specifications for Theory Engine

#### Input Specifications
* **Prompt Payload**:
  * `fact_sheet`: Case Fact Sheet JSON (Complainant, Accused, Property, Statutory Sections).
  * `subgraph_triples`: Top-30 verified entity relationships from Neo4j: `(EntityA)-[RELATION]->(EntityB)`.
  * `forensic_evidence_list`: Detected crime scene objects and vehicle ANPR matches with timestamps.
  * `historical_mo_matches`: Top-3 similar historical cases retrieved by Model 9.

#### Output Specifications
* **Theory Card JSON Schema**:
  ```json
  {
    "theory_title": "String",
    "rank": "Integer (1, 2, 3)",
    "confidence_score": "Float (0.0 to 1.0)",
    "primary_motive_category": "String enum",
    "ideological_context": "String enum",
    "suspected_syndicate": "String",
    "sub_events_chronology": [
      {
        "step": "Integer",
        "timestamp_window": "ISO 8601 String",
        "action": "String",
        "supporting_evidence": ["String array of citations"],
        "verified_entailment": "Boolean"
      }
    ],
    "unresolved_gaps_and_leads": [
      {
        "gap_description": "String",
        "linked_phantom_id": "String",
        "recommended_action": "String (Legal Statutory Next Step)"
      }
    ]
  }
  ```

---

### 5.3 NLI Citation-Validator (Anti-Hallucination Gate)
To prevent the LLM from fabricating false criminal allegations:
1. Every generated sentence $S_k$ is paired with its claimed evidence citation $E_k$.
2. A separate, frozen Natural Language Inference (NLI) classifier evaluates entailment:
   $$P(\text{Entailment} \mid E_k, S_k) \ge 0.85$$
3. If entailment fails, **the sentence is discarded and regenerated**. A sentence lacking a valid evidentiary citation is never presented to the investigator.

### 5.4 Motive, Ideology & Syndicate Profiling Engine
* **Motive Categories**: `FINANCIAL_GAIN`, `INTER_GANG_RIVALRY`, `TERROR_COMMUNAL_DESTABILIZATION`, `EXTORTION_RACKET`, `NARCO_DISTRIBUTION`.
* **Ideological Profiling**: Evaluates forensic markings (graffiti, pamphlet text from Model 1/2) and MO signatures against known extremist group operational manuals.
* **Mandatory Disclaimer**: Displayed with permanent banner: *"Investigative Hypothesis Only — Not a Judicial Finding."*

### 5.5 Suspicion-Strength Priority Lead Scoring Formulation
$$S(u) = 0.35 \cdot P_{\text{GNN}}(u) + 0.20 \cdot C_{\text{eigen}}(u) + 0.15 \cdot M_{\text{MO}}(u) + 0.20 \cdot E_{\text{direct}}(u) + 0.10 \cdot H_{\text{recidivism}}(u)$$

---

# 6. Database Layer: Physical Schemas & Storage Design

### 6.1 PostgreSQL System of Record (Complete DDL)

```sql
-- 1. Cases Table
CREATE TABLE cases (
    case_id VARCHAR(64) PRIMARY KEY,
    fir_number VARCHAR(64) NOT NULL,
    state VARCHAR(32) NOT NULL,
    district VARCHAR(64) NOT NULL,
    police_station VARCHAR(64) NOT NULL,
    incident_datetime TIMESTAMPTZ NOT NULL,
    triage_track VARCHAR(16) NOT NULL CHECK (triage_track IN ('TRACK_1_ROUTINE', 'TRACK_2_COMPLEX')),
    triage_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Master Entities Table
CREATE TABLE entities (
    entity_id VARCHAR(64) PRIMARY KEY,
    case_id VARCHAR(64) REFERENCES cases(case_id),
    entity_type VARCHAR(32) NOT NULL CHECK (entity_type IN ('PERSON', 'PHONE', 'VEHICLE', 'ACCOUNT', 'OBJECT', 'PHANTOM_ENTITY')),
    canonical_attributes JSONB NOT NULL,
    is_phantom BOOLEAN DEFAULT FALSE,
    phantom_lead_type VARCHAR(32),
    confidence_score NUMERIC(5, 4) NOT NULL,
    recommended_action TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Raw Evidentiary Edges Table
CREATE TABLE edges_raw (
    edge_id VARCHAR(64) PRIMARY KEY,
    case_id VARCHAR(64) REFERENCES cases(case_id),
    src_entity_id VARCHAR(64) REFERENCES entities(entity_id),
    dst_entity_id VARCHAR(64) REFERENCES entities(entity_id),
    edge_type VARCHAR(64) NOT NULL,
    confidence NUMERIC(5, 4) NOT NULL,
    source_evidence_ref VARCHAR(128) NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 4. Lead Lifecycle Tracking Table
CREATE TABLE lead_lifecycle (
    lead_id VARCHAR(64) PRIMARY KEY,
    entity_id VARCHAR(64) REFERENCES entities(entity_id),
    state VARCHAR(32) NOT NULL CHECK (state IN ('OPEN', 'DATA_REQUESTED', 'DATA_RECEIVED', 'RESOLVED', 'DISMISSED')),
    actor_id VARCHAR(64) NOT NULL,
    transition_reason TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 5. Cryptographic Audit Ledger (BSA 2023 §63 Compliant)
CREATE TABLE audit_events (
    event_id BIGSERIAL PRIMARY KEY,
    case_id VARCHAR(64) REFERENCES cases(case_id),
    actor_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    purpose_tag VARCHAR(128) NOT NULL,
    payload_hash_sha256 CHAR(64) NOT NULL,
    prev_hash_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 Neo4j Analytical Property Graph Schema
* **Node Labels**: `:Person`, `:Phone`, `:Vehicle`, `:FinancialAccount`, `:ObjectItem`, `:PhantomEntity`.
* **Relationship Types**: `[:CALLS]`, `[:OWNS]`, `[:TRANSACTS]`, `[:LINKED_TO]`, `[:CO_ACCUSED]`, `[:SUSPECTED_LINK_TO]`.
* **Sample Bounded Traversal Cypher Query**:
```cypher
MATCH path = (p:Person {entity_id: $suspect_id})-[:CALLS|OWNS|TRANSACTS*1..2]-(target)
WHERE all(r in relationships(path) WHERE r.confidence > 0.65)
RETURN path LIMIT 50;
```

---

# 7. Air-Gapped Infrastructure, Data Governance & Legal Compliance

### 7.1 Air-Gapped Zero-Leakage Architecture
1. **Network Layer Isolation**: Runs strictly within state police on-premise servers (State Data Centres - SDCs) or MeghRaj government cloud VPCs with no external internet routing.
2. **Local Model Execution**: PyTorch, Hugging Face transformers, and GGUF runtimes are pre-installed in self-contained Docker images.

### 7.2 BSA 2023 §63 Electronic Evidence Cryptographic Chaining
Under Section 63 of the Bharatiya Sakshya Adhiniyam, 2023 (BSA), electronic evidence requires complete provenance certification:
* Every uploaded file is digested via SHA-256 upon byte receipt:
  $$\mathbf{H}_t = \text{SHA256}(\mathbf{Payload}_t \,\|\, \mathbf{H}_{t-1} \,\|\, \text{Timestamp})$$
* Automated generation of §63 Certificates of Electronic Record detailing hash, capture timestamp, and verified model commit ID.

### 7.3 DPDP Act 2023 Compliance & Access Control
* **Statutory Criminal Investigation Exemption**: Invoked pursuant to Section 17 of the Digital Personal Data Protection Act, 2023.
* **Role-Based Access Control (RBAC)**: Enforced via Open Policy Agent (OPA) sidecars and PostgreSQL Row-Level Security (RLS). Constables cannot browse unassigned district intelligence.
* **Standing Bias Audit**: Monthly automated reporting monitoring false-positive disparity across demographic slices.

---

# 8. Traceability Matrix: Mapping Outline Stages to Technical Implementation

| Outline Stage | Conceptual Function | Concrete Technical Implementation in this System |
|:---|:---|:---|
| **Doc 1, Stage 1** | Ingestion & Adapters | Section 1.1 Unified Intake Console + N-source specific Python adapters |
| **Doc 1, Stage 2** | FIR Dual-Path Handling | Section 2.1 (CAS digital parser) & Section 2.2 (Qwen2-VL FIR KIE) |
| **Doc 1, Stage 3** | Extraction Models | Sections 2.3–2.9 (TrOCR, IndicBERT, Whisper, YOLO CCTV, ANPR, Co-DETR) |
| **Doc 1, Stage 4** | Entity Resolution & Graph | Section 2.11 (LaBSE transliteration) + PostgreSQL `entities` / `edges_raw` |
| **Doc 1, Stage 5** | Fact-Sheet Auto-Summarizer | Section 1.1 Stage 2 + Section 5.1 Structured bullet extractor |
| **Doc 1, Stage 6** | Triage Gate | Rule-based PostgreSQL service routing cases to Track 1 vs. Track 2 |
| **Doc 1, Stage 7** | Graph Analytics | Section 3.6 Classical algorithms (Louvain, PageRank, Betweenness) via Neo4j GDS |
| **Doc 1, Stage 8** | Lead & Hypothesis Chaining | Section 3.3 (Inductive HGT) + PostgreSQL `lead_lifecycle` table |
| **Doc 1, Stage 9** | Insight Generation | Section 5.1 Grounded LLM + Section 5.3 NLI Citation-Validator |
| **Doc 1, Stage 10**| Crime Reconstruction Engine | Section 5.1 Chronological sub-event synthesis + Motive/Ideology profiling |
| **Doc 1, Stage 11**| Data Governance & Audit | Section 7.2 (BSA 2023 §63 SHA-256 hash chaining) & Section 7.3 (DPDP/OPA) |
| **Doc 1, Stage 12**| Police System Integration | CCTNS/ICJS batch ETL adapters + e-Prisons co-incarceration graph edges |
| **Doc 1, Stage 13**| MLOps & Delta Pipeline | Delta event-driven re-computation + MLflow model registry + Airflow DAGs |
