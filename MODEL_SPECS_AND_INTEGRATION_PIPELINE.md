# SIH26189 — Master AI Model Specifications & Web Integration Pipeline

**AI-Powered Criminal Network Analysis & Crime Reconstruction System for Indian Police**
*Technical Model Specifications | API Contracts | Multimodal Ingestion Pipeline | GNN Historical Linkage | Air-Gapped Deployment*

---

## 1. System Vision & Unified Ingestion Architecture

### 1.1 The Operational Concept
The platform acts as a force multiplier for Indian police investigators by providing an **air-gapped, zero-cloud, multi-modal intelligence hub**. It transforms disparate, unstructured evidentiary artifacts (scanned FIRs, handwritten case diaries, intercepted/recorded phone calls, CCTV video feeds, car number plates, and crime scene photographs) into a unified, mathematically rigorous knowledge graph.

By comparing newly ingested case artifacts against historical crime databases (CCTNS records, previous charge-sheets, e-Prisons incarceration records, and telecom Call Detail Records), the system:
1. **Instantly surfaces key investigative summaries (Fact Sheets)** without manual transcription.
2. **Inductively links new suspects to historical criminal syndicates** using Graph Neural Networks (GNN).
3. **Spawns `PhantomEntity` nodes** for unidentified conspirators, burner numbers, and partial plates.
4. **Synthesizes ranked, evidence-cited crime reconstruction theories**, predicting the probable motive, ideological affiliation, and recommended investigative leads.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED EVIDENCE INTAKE CONSOLE (Single Drop-Zone)                         │
│   Upload any file: .pdf, .jpg, .png, .mp4, .wav, .mp3, .csv (CDR / Tower Dump), .json (CCTNS)   │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: AUTO-ROUTER & MULTIMODAL EXTRACTION                           │
├───────────────────────┬───────────────────────┬─────────────────────────┬───────────────────────┤
│  Scanned / Image FIR  │ Handwritten Annexures │  Audio / Dispatch Call  │   CCTV / Video Feed   │
│     [Model 1: KIE]    │    [Model 2: TrOCR]   │     [Model 4: ASR]      │  [Model 5: YOLO CCTV] │
│     Qwen2-VL-2B LoRA  │   Indic-TrOCR Seq2Seq │   Whisper-Small Indic   │   8-Class Surveillance│
├───────────────────────┴───────────────────────┴─────────────────────────┴───────────────────────┤
│  Vehicle Plate Image  │  Crime Scene Images   │  Unstructured Case Text │ Structured CDR / Bank │
│     [Model 6: ANPR]   │   [Model 7: Vision]   │     [Model 3: NER]      │   Rule / GBM Parsers  │
│    YOLOv8 + CRNN CTC  │   Crime Scene Co-DETR │   IndicBERT-v2 + CRF    │   Structuring Detector│
└───────────────────────┴───────────────────────┴─────────────────────────┴───────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 2: FACT-SHEET SUMMARIZER & PRESENTATION                          │
│   - Immediately extracts: Who, What, When, Where, Evidence on File, Stolen Property, Violations │
│   - Performs automated Vahan RTO lookup on detected vehicle plates                              │
│   - Displays immediate Fact-Sheet Diff to Investigator on Web UI Dashboard                      │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: ENTITY RESOLUTION & HYBRID GRAPH SYNC                             │
│   - Deterministic Regex Matching (Phone numbers, Plates, IMEIs, Bank A/Cs)                      │
│   - Transliteration-Aware Fuzzy Matcher (IndicBERT/LaBSE embeddings for names/addresses)       │
│   - System of Record: PostgreSQL (Immutable Append-Only Audit Ledger, BSA 2023 §63 Compliant)  │
│   - Analytical Graph: Neo4j (Property Graph Model with temporal validity windows)               │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│           STAGE 4: INDUCTIVE GNN LAYER — CONNECTING NEW DATA TO HISTORICAL RECORDS              │
│                       [Model 8: Heterogeneous Graph Transformer (HGT)]                          │
│   - Checkpoint: hgt_stageB_checkpoint.pt (Pretrained + Supervised Fine-Tuned)                  │
│   - Projects new case entities into historical syndicate multi-relational graph                 │
│   - Zero-Retrain Inductive Inference: Embeds newly created nodes via neighbor sampling          │
│   - Link Prediction: Evaluates P(suspected_link_to) between new suspects & historical bosses    │
│   - Uncovers PhantomEntity nodes (Unknown callers, partial plates, getaway drivers)             │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│            STAGE 5: CRIME RECONSTRUCTION, MOTIVE & INVESTIGATIVE THEORY ENGINE                  │
│               [Model 9: Serial Crime MO-Engine] & [Model 10: Grounded Theory LLM]               │
│   - Matches Modus Operandi (MO) against unsolved historical case bank via Metric Learning       │
│   - Generates Ranked Crime Hypotheses: Step-by-step chronology with per-sentence citations     │
│   - Profiles Motive & Ideology: Analyzes organizational patterns, extremist signatures          │
│   - Priority Lead Ranking: Identifies primary conspirators with highest suspicion strength      │
│   - Enforces NLI Citation Validator: Automatically drops any ungrounded hallucination           │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               STAGE 6: INVESTIGATOR VISUAL WORKSPACE                            │
│   - Interactive Canvas: Cytoscape.js visual graph with distinct dashed lines for GNN hypothesis │
│   - Kanban Lead Board: Tracks PhantomEntity lifecycle (Open -> Data Requested -> Resolved)     │
│   - Theory & Motive Dossier: Formatted court-admissible briefing export                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Inventory & Checkpoint Registry

The architecture deploys 10 dedicated AI/ML models across the ingestion, graph reasoning, and reconstruction lifecycle.

| # | Model Name | Primary Task | Base Architecture | Checkpoint / Artifact Path | Model Size | Parameter Count | VRAM / Latency |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **M1** | **FIR KIE** | Structured extraction from scanned IIF-1 FIRs | `Qwen2-VL-2B-Instruct` | `qwen2_vl_fir_kie_final/adapter_model.safetensors` | 140 MB (LoRA) | 2.21B Total / 36.9M Trainable | 4.5 GB / 1.4s |
| **M2** | **Handwriting OCR** | Scanned diary entries & witness statements | `TrOCR-Large` / `Indic-TrOCR` | `trocr_indic_handwriting/pytorch_model.bin` | 1.3 GB | 334M Total | 2.1 GB / 850ms |
| **M3** | **Indic Legal NER** | Named entity extraction from case narratives | `IndicBERT-v2-MLM` + CRF | `indicbert_legal_ner/pytorch_model.bin` | 540 MB | 278M Total | 1.2 GB / 45ms |
| **M4** | **Indic ASR** | Dispatch calls & wiretaps under heavy noise | `whisper-small` / `indicwhisper` | `RESULTS/Audio_Transcription/adapter_model.safetensors` | 28.4 MB (LoRA) | 241.8M Total / 3.5M Trainable | 1.8 GB / 350ms per 30s |
| **M5** | **CCTV Surveillance** | Traffic/street vehicle & person detection | `YOLOv8s` (8-Class Master Taxonomy) | `RESULTS/Audio_Transcription/best.pt` / `yolo_cctv.pt` | 22.5 MB | 11.2M Total | 1.1 GB / 7.5ms (133 FPS) |
| **M6** | **ANPR & OCR** | License plate localization & text reading | `YOLOv8n` + 7-Layer CNN BiLSTM CTC | `RESULTS/No. Plate/best.pt` & `best_crnn.pt` | 24.5 MB + 34.9 MB | 3.2M + 8.7M Total | 1.2 GB / 25ms end-to-end |
| **M7** | **Crime Scene Vision** | Weapon, contraband & forensic object detection | `Co-DETR` / `YOLOv8x-World` | `crime_scene_detector/best.pt` | 136 MB | 68.2M Total | 2.8 GB / 42ms |
| **M8** | **Criminal GNN** | Inductive link prediction & phantom embedding | `Heterogeneous Graph Transformer (HGT)` | `RESULTS/GNN/hgt_stageB_checkpoint.pt` | 3.82 MB | 1.4M Total | 1.5 GB / 18ms per 10k nodes |
| **M9** | **Serial Crime MO** | Modus Operandi similarity metric learning | Siamese Triplet Dense Network | `mo_matcher/siamese_triplet.pt` | 12.4 MB | 3.1M Total | 0.4 GB / 8ms per query |
| **M10**| **Reconstruction LLM**| Multi-hypothesis narrative theory generation | `Qwen2.5-7B-Instruct-GGUF` (Q4_K_M) + NLI | `reconstruction_llm/qwen2.5_7b_q4.gguf` | 4.6 GB | 7.6B Total | 5.8 GB / 2.2s per theory |

---

## 3. Exhaustive Model Specifications

### Model 1: Multimodal FIR Key Information Extraction (Qwen2-VL-2B-Instruct)
* **Base Model**: `Qwen/Qwen2-VL-2B-Instruct` (Multimodal Vision-Language Model).
* **Architecture**:
  * **Vision Encoder**: Dynamic resolution ViT with 2D Rotary Position Embeddings (2D-RoPE). Images are processed in native aspect ratios using $28 \times 28$ pixel spatial patches.
  * **Cross-Modal Projector**: Linear projection layer mapping ViT token representations into the decoder hidden dimension.
  * **Language Decoder**: 28-layer Transformer Decoder featuring Grouped-Query Attention (GQA) and 152,000 token vocabulary natively supporting Indian scripts.
* **Fine-Tuning Method**: LoRA ($r=32$, $\alpha=32$, dropout $0.05$) applied across `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **Vision Backbone State**: 100% frozen to preserve edge fidelity for official stamps, signatures, and table borders.
* **Input Specs**: Image or single-page PDF up to $1280 \times 1280$ pixels.
* **Output Specs**: Structured JSON mapping the national CCTNS IIF-1 standard (Acts, Sections, Complainant, Accused, Incidental Entities, Property).
* **Hardware Footprint**: 4.5 GB VRAM in `bfloat16`, 1.4 seconds latency on T4/RTX 3060.

### Model 2: Handwritten Document & Annexure OCR (TrOCR-Large / Indic-TrOCR)
* **Base Model**: `microsoft/trocr-large-handwritten` adapted for Indian scripts.
* **Architecture**:
  * **Encoder**: Vision Transformer (ViT) dividing handwritten page patches into sequence representations.
  * **Decoder**: Autoregressive Transformer text decoder decoding text line-by-line.
* **Target Data**: Police case diaries, handwritten witness interrogation transcripts (Section 161/164 CrPC / BNSS equivalents).
* **Input Specs**: Grayscale/RGB line-level crops ($384 \times 384$) or full scanned diary pages.
* **Output Specs**: Raw UTF-8 text transcript with line-by-line bounding coordinates and character-level confidence scores.
* **Hardware Footprint**: 2.1 GB VRAM in float16, 850 ms per page.

### Model 3: Multilingual / Indic Legal NER (IndicBERT-v2 / MuRIL + CRF)
* **Base Model**: `ai4bharat/indic-bert` or `google/muril-base-cased`.
* **Architecture**: 12-layer Bidirectional Transformer Encoder with a Linear-Chain Conditional Random Field (CRF) sequence labeling head.
* **Taxonomy (8 Entity Classes)**: `B/I-PERSON`, `B/I-PHONE`, `B/I-VEHICLE_PLATE`, `B/I-ADDRESS`, `B/I-ORG`, `B/I-DATE`, `B/I-MONEY_AMOUNT`, `B/I-WEAPON_ITEM`.
* **Input Specs**: UTF-8 text string up to 512 tokens (supports Devanagari, Gurmukhi, Bengali, Tamil, Telugu, and Romanized transliteration).
* **Output Specs**: Span-level entity extraction tuples `(entity_text, label, start_char, end_char, confidence)`.
* **Hardware Footprint**: 1.2 GB VRAM, 45 ms latency per 500-word statement.

### Model 4: Audio Dispatch & Intercept Transcription (Whisper-Small Indic + Noise LoRA)
* **Base Model**: `openai/whisper-small` / `ai4bharat/indicwhisper-small`.
* **Architecture**: 12-layer Encoder (80-channel log-mel spectrograms) + 12-layer Decoder.
* **Trained LoRA Weights**: `RESULTS/Audio_Transcription/adapter_model.safetensors` ($28.4\text{ MB}$).
* **Adaptation**: LoRA ($r=32$, $\alpha=64$, dropout $0.05$) on `q_proj`, `k_proj`, `v_proj`, `out_proj`.
* **Novelty Training Pipeline**:
  * Telephone bandpass filtering ($300 - 3400\text{ Hz}$).
  * GSM full-rate (06.10) / AMR-NB codec round-trip compression simulation.
  * Variable SNR noise injection ($5 - 20\text{ dB}$) using sirens, road traffic, and police radio static.
  * **Entity-WER Optimization**: Specifically minimizing error rates on numbers, names, and plate mentions.
* **Input Specs**: 16 kHz mono WAV/MP3/AMR audio stream.
* **Output Specs**: Transcribed text, language ID, and regex-extracted investigative entities.
* **Hardware Footprint**: 1.8 GB VRAM, 350 ms per 30-second audio chunk.

### Model 5: CCTV Traffic & Street Surveillance (YOLOv8 8-Class Master Taxonomy)
* **Base Model**: YOLOv8s customized for Indian urban surveillance.
* **Weights**: `RESULTS/Audio_Transcription/best.pt` / `yolo_cctv.pt` ($22.5\text{ MB}$).
* **Unified Master Taxonomy (8 Classes)**: `0: person`, `1: car`, `2: truck`, `3: bus`, `4: motorcycle`, `5: bicycle`, `6: autorickshaw`, `7: van`.
* **Domain Adaptation**: Trained against synthetic CCTV degradations (low-light Poisson sensor noise, IR camera CLAHE mixing, JPEG block artifacts, downscale-upscale blur, and high-angle perspective transforms).
* **Input Specs**: 1080p RTSP stream or JPEG image ($640 \times 640$, $416 \times 416$, or $320 \times 320$).
* **Output Specs**: Bounding boxes `[x1, y1, x2, y2, confidence, class_id]`.
* **Hardware Footprint**: 1.1 GB VRAM, 7.5 ms per frame ($133\text{ FPS}$).

### Model 6: ANPR License Plate Reader & RTO Decoder (YOLOv8 + CRNN-BiLSTM-CTC)
* **Architecture**: Two-stage cascaded neural pipeline.
  * **Stage A (Detector)**: YOLOv8n plate detector (`RESULTS/No. Plate/best.pt`, 24.5 MB, $\text{mAP}_{50} = 0.9439$).
  * **Stage B (Reader)**: 7-layer CNN feature extractor + 2-layer Bidirectional LSTM ($8.7\text{M}$ params, `RESULTS/No. Plate/best_crnn.pt`, 34.9 MB).
* **Decoding & RTO Ambiguity Resolver**:
  * CTC Beam Search (Width $B=10$, top-3 candidates with log-probabilities).
  * Lexicon-constrained correction checking the first 4 characters against India's **1,370+ legal RTO codes** (e.g. `DL01` to `DL13`, `MH01` to `MH53`).
  * Confusion matrix disambiguating `0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `I`, `8` $\leftrightarrow$ `B`, `5` $\leftrightarrow$ `S`.
  * Partial Plate Flagging (`confidence < 0.60` creates a `PhantomEntity`).
* **Verified Test Metrics**: **$98.08\%$ exact match accuracy**, **$99.44\%$ RTO prefix accuracy**.
* **Input Specs**: Full vehicle photo or cropped plate image.
* **Output Specs**: Normalized plate string + Vahan registration dossier (Maker, Model, Fuel, Color, Owner Mask, RTO Office).
* **Hardware Footprint**: 1.2 GB VRAM, 25 ms end-to-end latency.

### Model 7: Forensic Crime Scene Object & Physical Evidence Analyzer (YOLOv8x / Co-DETR)
* **Base Architecture**: Co-DETR or YOLOv8x fine-tuned on forensic scene evidence.
* **Forensic Class Taxonomy (18 Classes)**:
  * *Weapons/Ballistics*: `firearm_handgun`, `firearm_rifle`, `knife_blade`, `spent_cartridge`, `bullet_lead`, `holster`.
  * *Explosives/Contraband*: `ied_timer`, `battery_pack`, `detonator_wire`, `chemical_canister`, `narcotic_packet`.
  * *Burglary/Forensics*: `crowbar`, `lockpick`, `broken_lock`, `footwear_impression`, `blood_spatter`, `fingerprint_powder_lift`.
  * *Vehicle Tampering*: `broken_ignition`, `tampered_vin_plate`.
* **Input Specs**: High-resolution crime scene photos ($1280 \times 1280$).
* **Output Specs**: Bounding boxes, forensic object class, confidence, and MO feature tags (e.g. `forced_entry_technique`, `improvised_initiator`).
* **Hardware Footprint**: 2.8 GB VRAM, 42 ms per image.

### Model 8: Criminal Network Reasoning & Link Prediction (Heterogeneous Graph Transformer - HGT)
* **Architecture**: 2-layer Heterogeneous Graph Transformer (`torch_geometric.nn.HGTConv`).
* **Trained Weights**: `RESULTS/GNN/hgt_stageB_checkpoint.pt` ($3.82\text{ MB}$).
* **Node Types & Dimensions**:
  * `person` ($d=32$): Name embedding proxy + criminal history multi-hot + degree + recency.
  * `phone` ($d=3$): IMEI persistence + activity recency + burner churn rate.
  * `financial_account` ($d=8$): Transaction velocity, credit/debit ratio, dormancy burst, structuring alert flag.
  * `object` ($d=17$): 5-dim one-hot type + 12-dim MO signature vector.
  * `phantom_entity` ($d=16$): 4-dim lead category + 12-dim sparse partial feature vector.
* **Edge Relations (14 Canonical & Reverse Directed Types)**:
  * `('person', 'calls', 'person')`, `('person', 'suspected_link_to', 'person')`, `('person', 'owns', 'phone')`, `('person', 'owns', 'financial_account')`, `('financial_account', 'transacts', 'financial_account')`, `('person', 'linked_to', 'object')`, `('phantom_entity', 'partial_match', 'person')` + 7 reverse edge relations.
* **Link Predictor**: Bilinear Concatenation MLP: $\hat{y} = \sigma(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 [\mathbf{h}_u \,\|\, \mathbf{h}_v] + \mathbf{b}_1) + b_2)$.
* **Inductive Guarantee**: Embeds brand-new `PhantomEntity` nodes in a single forward pass with **zero retraining**.
* **Verified Metrics**: Pretrain test AUROC $0.8833$, AUPRC $0.5034$; consistently beats Adamic-Adar on scale-free criminal graphs.
* **Hardware Footprint**: 1.5 GB VRAM (or runs on CPU in $<500\text{ MB}$ RAM), 18 ms per 10k nodes.

### Model 9: Serial Crime MO-Similarity Metric Engine (Siamese Triplet Network + FAISS)
* **Architecture**: Dense Deep Metric Network ($256 \to 128 \to 64$) mapping multi-modal case vectors into a unified Modus Operandi metric space.
* **Input Feature Vector ($d=48$)**:
  * Geospatial cyclical encoding (Lat/Long distance to transport hubs) ($d=4$).
  * Temporal cyclical encoding (time of day, day of week, seasonal offset) ($d=4$).
  * Method of Commission text embedding (from IndicBERT) ($d=32$).
  * Physical Object signature (from Model 7 Crime Scene Vision) ($d=8$).
* **Loss Function**: Triplet Margin Loss with online semi-hard negative mining ($\text{margin} = 0.3$).
* **Output Specs**: 64-dim L2-normalized MO embedding; sub-millisecond Top-$K$ retrieval against historical unsolved cases via FAISS HNSW.
* **Hardware Footprint**: 0.4 GB VRAM / CPU, 8 ms per query.

### Model 10: Crime Reconstruction, Motive & Narrative Theory Engine (Self-Hosted LLM + NLI Validator)
* **Base Model**: `Qwen2.5-7B-Instruct` quantized to 4-bit (`Q4_K_M` GGUF, $4.6\text{ GB}$).
* **Fine-Tuning**: LoRA instruction-tuned on structured fact sheets, timelines, and case briefs.
* **Dual-Component Architecture**:
  1. **Narrative Generator**: Synthesizes chronological sub-events, motive category, ideological affiliation, and investigative gaps.
  2. **NLI Citation-Validator**: A lightweight entailment classifier (`RoBERTa-large-MNLI` / `Indic-NLI`). Any sentence that cannot be formally entailed from the retrieved evidence graph is **automatically purged and regenerated**.
* **Hardware Footprint**: 5.8 GB VRAM on GPU, 2.2 seconds latency per generated theory.

---

## 4. The System Design: GNN as the Connecting Layer between New Data and Historical Records

A critical architectural challenge is: **How does a newly uploaded file connect with 5 years of historical CCTNS crime records?**

The Heterogeneous Graph Transformer (Model 8) acts as the active **inductive reasoning layer** bridging newly observed data with historical graphs.

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

## 5. PhantomEntity Identification, Conspiracy Detection & Perpetrator Ranking

### 5.1 What is a PhantomEntity?
When physical evidence indicates that a person, vehicle, or account was an active participant in an incident, but their legal identity remains unresolved, the system **never drops the signal**. It instantiates a `PhantomEntity` node.

#### Common Phantom Archetypes
1. **Unidentified Caller / Burner (`lead_type = phone_unresolved`)**: A number calling the accused multiple times immediately before an incident, purchased with fraudulent KYC.
2. **Partial Plate / Getaway Vehicle (`lead_type = partial_plate`)**: A vehicle captured on CCTV with an obscured number plate (e.g. `DL07??4891`).
3. **Unidentified Accomplice / Face (`lead_type = face_unidentified`)**: An accomplice visible in crime scene CCTV or drone footage lacking an immediate NAFIS/AFRS biometric match.
4. **Proxy Money-Mule Account (`lead_type = financial_mule`)**: An account receiving rapid pass-through deposits structured to evade PMLA reporting thresholds.

### 5.2 Priority Lead & Suspicion-Strength Ranking Formula
To assist the investigator in prioritizing which suspect or phantom node to pursue first, the system computes a multi-factorial **Suspicion-Strength Score** $S(u) \in [0, 1]$:

$$S(u) = w_1 \cdot P_{\text{GNN}}(u, \text{Crime}) + w_2 \cdot C_{\text{eigen}}(u) + w_3 \cdot M_{\text{MO}}(u) + w_4 \cdot E_{\text{direct}}(u) + w_5 \cdot H_{\text{recidivism}}(u)$$

Where:
* $P_{\text{GNN}}(u, \text{Crime})$: Model 8 HGT link prediction probability between suspect $u$ and the current incident node ($w_1 = 0.35$).
* $C_{\text{eigen}}(u)$: Eigenvector / betweenness centrality in the criminal syndicate graph ($w_2 = 0.20$).
* $M_{\text{MO}}(u)$: Model 9 Modus Operandi metric similarity to historical case signatures ($w_3 = 0.15$).
* $E_{\text{direct}}(u)$: Normalized weight of direct forensic evidence (fingerprint, ballistic, CCTV, CDR co-location) ($w_4 = 0.20$).
* $H_{\text{recidivism}}(u)$: Prior charge-sheet count under matching statutory acts ($w_5 = 0.10$).

*Mandatory Legal Notice*: On every UI surface and export, this score is explicitly displayed with the disclaimer: **"Investigative Prioritization Score — Not a Legal Determination of Guilt."**

---

## 6. Crime Reconstruction, Ideology/Motive Analysis & Theory Engine

### 6.1 Multi-Hypothesis Chronological Reconstruction
Model 10 consumes the structured Fact Sheet, the HGT graph embeddings, and the crime scene object detections to generate **2 to 3 ranked, competing theories** of how the crime occurred.

#### Theory Card Schema
```json
{
  "theory_id": "TH-001",
  "rank": 1,
  "confidence_score": 0.842,
  "theory_title": "Targeted Syndicate Vehicle Theft for Inter-State Contraband Transit",
  "motive_and_ideology": {
    "primary_motive_category": "ORGANIZED_PROPERTY_CRIME_LOGISTICS",
    "ideological_context": "NON_IDEOLOGICAL / PROFIT_DRIVEN_SYNDICATE",
    "suspected_syndicate": "Mustaqeem Inter-State Auto-Lifting Network",
    "modus_operandi_pattern": "Tampering ignition via specialized OBD decoders, transit within 6 hours across NCR borders to Western UP dismantling hubs."
  },
  "sub_events_chronology": [
    {
      "step": 1,
      "timestamp_window": "2026-03-12T13:45:00 to 14:00:00",
      "action": "Suspect Irfan @ Chhotu arrived at Red Fort parking lot on foot.",
      "supporting_evidence": ["CCTV Camera 4 frame 1420", "Witness statement of parking attendant W-1"],
      "verified_entailment": true
    },
    {
      "step": 2,
      "timestamp_window": "2026-03-12T14:05:00",
      "action": "Forced entry into Maruti Swift (DL01AB1234) using lock-bypass tool.",
      "supporting_evidence": ["Model 7 detected lockpick marks on driver door latch in Photo CS-04"],
      "verified_entailment": true
    },
    {
      "step": 3,
      "timestamp_window": "2026-03-12T14:12:00",
      "action": "Vehicle exited parking lot towards Kashmere Gate ISBT, coordinating with Unknown Caller via phone 9871987654.",
      "supporting_evidence": ["ANPR Detection Gate 2 at 14:12:45", "CDR Tower Dump cell 401A"],
      "verified_entailment": true
    }
  ],
  "unresolved_gaps_and_leads": [
    {
      "gap_description": "Identity of handler communicating via burner phone 9871987654 during transit.",
      "linked_phantom_id": "PH-089",
      "recommended_action": "Serve Section 5(2) Telegraph Act notice to telecom provider for CDR cell tower path."
    }
  ]
}
```

---

## 7. API Gateway Contracts & Endpoint Specifications

### 7.1 Unified Evidence Intake Endpoint
```http
POST /api/v1/evidence/upload
Content-Type: multipart/form-data
```
#### Request Form Data
| Parameter | Type | Required | Description |
|:---|:---|:---|:---|
| `case_id` | String | Yes | Unique case identifier (e.g. `"FIR-2026-DEL-0142"`). |
| `file` | Binary | Yes | The evidence file (.pdf, .jpg, .png, .wav, .mp3, .mp4, .csv). |
| `evidence_category` | String | No | Auto-detected if omitted (`"DOCUMENT"`, `"AUDIO"`, `"CCTV"`, `"PLATE"`, `"CRIME_SCENE"`). |
| `recorded_datetime` | String | No | ISO 8601 timestamp of evidence capture. |

#### Response JSON
```json
{
  "status": "success",
  "evidence_id": "EV-2026-0941",
  "file_hash_sha256": "8f4a2b91c3e4...",
  "bsa_certificate_id": "BSA-63-CERT-0142-0941",
  "routed_pipeline": "CRIME_SCENE_VISION_PIPELINE",
  "fact_sheet_update": {
    "new_objects_found": ["spent_cartridge", "forced_lock"],
    "new_entities_extracted": []
  },
  "graph_updates": {
    "nodes_created": 2,
    "edges_created": 3,
    "phantom_entities_spawned": 1
  }
}
```

### 7.2 Run GNN Criminal Link Prediction
```http
POST /api/v1/graph/predict-conspiracy
Content-Type: application/json
```
#### Request Body
```json
{
  "case_id": "FIR-2026-DEL-0142",
  "target_entity_id": "ENT-1042",
  "max_hops": 2,
  "confidence_threshold": 0.65
}
```
#### Response JSON
```json
{
  "status": "success",
  "target_entity": {"id": "ENT-1042", "name": "Irfan @ Chhotu"},
  "predicted_conspirators": [
    {
      "entity_id": "ENT-89",
      "name": "Mustaqeem Syndicate Boss",
      "link_probability": 0.8914,
      "relationship_type": "suspected_link_to",
      "supporting_graph_motifs": [
        "Common phone contact within 1 hour of incident",
        "Matching MO signature in Western UP auto-thefts"
      ],
      "recommended_action": "Interrogate suspect regarding handlers in Mustaqeem syndicate."
    }
  ]
}
```

### 7.3 Generate Crime Theories & Motive Analysis
```http
POST /api/v1/reconstruction/generate-theories
Content-Type: application/json
```
#### Request Body
```json
{
  "case_id": "FIR-2026-DEL-0142",
  "num_hypotheses": 3,
  "enforce_nli_validation": true
}
```
#### Response JSON
Returns an array of validated theory cards conforming to the schema in [Section 6.1](#61-multi-hypothesis-chronological-reconstruction).

---

## 8. Hardware Sizing, Air-Gapped Security & DPDP Compliance

### 8.1 Hardware Allocation Matrix (Air-Gapped Workstation / Server)

| Tier | Component | Allocation | Concurrency Mode |
|:---|:---|:---|:---|
| **VRAM Budget** | Model 1 (Qwen2-VL FIR KIE) | 4.5 GB VRAM | On-Demand Worker |
| | Model 4 (Whisper Indic ASR) | 1.8 GB VRAM | On-Demand Worker |
| | Model 5 & 6 (YOLO CCTV + ANPR) | 2.3 GB VRAM | Continuous Video Worker |
| | Model 7 (Crime Scene Co-DETR) | 2.8 GB VRAM | On-Demand Worker |
| | Model 8 (Heterogeneous GNN) | 1.5 GB VRAM | CPU / GPU Shared |
| | Model 10 (Reconstruction LLM) | 5.8 GB VRAM | Case-Triggered Queue |
| **Total Hardware** | **Single 16GB GPU Node** | **15.2 GB Peak** | **1x NVIDIA RTX 4080 (16GB) or Tesla T4 (16GB)** |
| **System RAM** | PostgreSQL + Neo4j + MinIO | 32 GB RAM | 8 CPU Cores |

### 8.2 Air-Gapped Zero-Leakage Guarantee
* **No Outbound Network Sockets**: The application network is isolated via Docker internal bridge networks with `internal: true`.
* **Self-Contained Model Weights**: All 10 model weights, tokenizer files, vocabularies, and RTO lexicon tables are baked into local image volumes. Zero calls to Hugging Face, OpenAI, or external endpoints.

### 8.3 Digital Personal Data Protection (DPDP) Act 2023 Compliance
* **Statutory Investigation Exemption**: Data processing is strictly scoped to authorized criminal investigations under statutory provisions (BNSS / CrPC).
* **Role-Based Access Control (RBAC/ABAC)**: Enforced via Open Policy Agent (OPA). A station constable cannot view state crime-branch organized crime intelligence without written authorization.
* **Cryptographic BSA 2023 §63 Certificates**: Every uploaded evidence file is signed with an SHA-256 hash and appended to an immutable append-only ledger before any AI model touches it.

---

## 9. Production Docker Compose & Deployment Stack

Save as `docker-compose.yml` to launch the complete offline intelligence suite:

```yaml
version: '3.8'

networks:
  sih_internal:
    driver: bridge
    internal: true  # Guarantees 100% air-gapped zero internet access

services:
  postgres:
    image: postgres:16-alpine
    container_name: sih_postgres
    restart: always
    networks:
      - sih_internal
    environment:
      POSTGRES_DB: sih_police_db
      POSTGRES_USER: sih_admin
      POSTGRES_PASSWORD: secure_police_password_2026
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  neo4j:
    image: neo4j:5.18-community
    container_name: sih_neo4j
    restart: always
    networks:
      - sih_internal
    environment:
      NEO4J_AUTH: neo4j/secure_graph_password_2026
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"

  minio:
    image: minio/minio:RELEASE.2024-03-05T04-48-44Z
    container_name: sih_minio
    restart: always
    networks:
      - sih_internal
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio_police_admin
      MINIO_ROOT_PASSWORD: minio_police_secret_2026
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  ai_inference_workers:
    build:
      context: .
      dockerfile: Dockerfile.ai
    container_name: sih_ai_inference
    restart: always
    networks:
      - sih_internal
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - TORCH_DEVICE=cuda
      - CHECKPOINTS_ROOT=/app/RESULTS
    volumes:
      - ./RESULTS:/app/RESULTS:ro
    ports:
      - "8001:8001"

  api_gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    container_name: sih_gateway
    restart: always
    networks:
      - sih_internal
    depends_on:
      - postgres
      - neo4j
      - minio
      - ai_inference_workers
    environment:
      - DATABASE_URL=postgresql://sih_admin:secure_police_password_2026@postgres:5432/sih_police_db
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=secure_graph_password_2026
      - MINIO_ENDPOINT=minio:9000
      - AI_ENGINE_URL=http://ai_inference_workers:8001
    ports:
      - "8000:8000"

volumes:
  postgres_data:
  neo4j_data:
  minio_data:
```
