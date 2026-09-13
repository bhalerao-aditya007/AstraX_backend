**SIH26189**

**Complete Outline**

**AI-Powered Criminal Network Analysis System for Indian Police**

*Technical Architecture & Data Pipeline • Input/Output Specification • User Interface Design*

# **DOCUMENT 1 — Technical Architecture & Data Pipeline**

## **1. Guiding architectural principles**

Every design decision below is filtered through one question: **does this survive contact with a real, under-resourced, 16,000+-station, multi-state, multilingual police system with existing legacy infrastructure (CCTNS) that cannot be replaced, only integrated with?** A design that requires all data to be clean, all stations to be fully digitized, or all case data to flow through one central pipeline in real time will fail on contact with reality. The architecture assumes messiness, partial data, intermittent connectivity, and legacy coexistence as permanent conditions, not transitional problems.

A second principle sits alongside the first: **the system's job is not "does a match exist," it's "what is the next question an investigator should ask."** No module is permitted to dead-end silently. An unresolved suspect, an unidentified caller, an unmatched face — all of these become tracked, actionable leads with a concrete recommended next step, never a blank screen or a "no result found."

A third principle governs every automated judgment the system makes about a person: **every automated inference — including motive, ideology, and organizational-affiliation hypotheses — carries the same evidentiary discipline as every other output in this system: citation, confidence, and standing bias oversight.** Automation of a judgment does not exempt it from the system's core "hypothesis, not finding" contract; if anything it raises the bar, because there is no human in the loop at generation time to catch an error before it reaches an investigator's screen.

**Deployment constraint, non-negotiable:** the entire system is **air-gapped**. No case data, derived embedding, model weight, log, or inference call ever leaves the government-owned perimeter (state data center or NIC/MeghRaj-class government cloud). No third-party hosted API is used anywhere in the pipeline, including for the LLM stack that drives insight generation, crime reconstruction, and motive/ideology hypothesis generation. Every model in the system is self-hostable and runs entirely on infrastructure the department controls. DPDP Act 2023 purpose-limitation and audit obligations, and BSA 2023 §63 evidentiary certification, apply uniformly to every module.

---

## **2. Stage 1 — Ingestion & adapter layer**

**Core decision: one internal normalized schema, N source-specific adapters.** Every source system — CCTNS export, e-FIR portal, CDR feed, NAFIS/AFRS match result, e-Prisons record, NCRP complaint, Vahan/Saarthi lookup, FIU-IND STR — gets its own thin adapter microservice that translates its native format into the internal entity/event schema (Document 2, Section 3). Nothing downstream ever talks to a source system directly. This is the decision that makes "swap demo data for real CCTNS data" an adapter-writing exercise rather than a redesign, and it is also what lets a new state's CAS instance or a new telecom provider's CDR format be onboarded without touching the core pipeline.

**Best practices at this stage:**

* **Idempotent ingestion.** Re-ingesting the same FIR twice must never create duplicate nodes — every ingested record carries a source-system ID plus a content hash, checked against a dedup index before processing begins.
* **Schema versioning.** Source systems change their export formats over time (CCTNS itself has evolved). Every adapter carries a version tag for its expected input schema and fails loudly, never silently, on unexpected format drift — silent misparsing of an FIR field is a far worse failure than a visible ingestion error.
* **Batch vs. streaming, chosen per source, never globally.** CCTNS, e-Prisons, and Vahan data realistically arrive as periodic batch exports — build scheduled batch ETL for these. CDR and tower-dump requests are case-specific, on-demand pulls, not a continuous stream. Do not over-engineer a real-time streaming architecture (Kafka-class infrastructure) for source systems that don't produce real-time data — that complexity buys nothing and adds failure surface.
* **Offline-first field capture.** For stations with poor connectivity, the FIR-image-capture app must queue captures locally and sync when connectivity returns — an officer's workflow is never blocked on network availability.
* **Event-driven delta path.** Every ingestion event — including a single new evidence item arriving weeks or months into an already-open case — is treated as its own discrete unit of work that triggers a scoped downstream re-run through the pipeline, rather than sitting idle until the next scheduled batch cycle.

### **2.1 Stage 1.5 — Unified Evidence Intake Console**

The Unified Evidence Intake Console is a **single UI/API surface for unstructured, ad-hoc evidence** — scanned/photographed FIRs, handwritten annexures, crime-scene photographs, CCTV frame/video exports, vehicle plate photos, and audio recordings (dispatch, statements, intercepts). Structured registry data — CCTNS/CAS, Vahan/Saarthi, e-Prisons, CDR, FIU-IND STRs — continues to arrive via scheduled batch adapters or case-specific, warrant-gated on-demand lookups, exactly as in Stage 12, never as a generic upload. CDR/tower-dump/interception data is always a case-specific, warrant-triggered pull; Vahan/e-Prisons/CCTNS are batch or lookup adapters. Keeping this line explicit prevents two real failure modes: an investigator manually re-uploading data that already has a lawful, auditable pull mechanism, and an accidental side-door for bulk/unwarranted structured-data ingestion.

**What the Console does:**

1. **Single drop point, format-routed automatically.** An investigator or station officer drags in a file (or the mobile capture app submits one); the Console fingerprints the file's actual content type (image / PDF / audio / video) and routes it to the correct extraction model without the user needing to know or select which model applies.
2. **Immediate ingestion receipt.** File hash, BSA 2023 §63 evidence certificate, and a queued/processing/complete status are generated the instant the file lands — no blocking spinners, a visible sync status at all times, even on the desktop-primary investigator console.
3. **Routing table:**

   | Uploaded file type | Routed to |
   | :--- | :--- |
   | Scanned/photographed FIR (typed) | Model 1 — FIR Key Information Extraction (Qwen2-VL-2B) |
   | Scanned/photographed handwritten annexure | Handwritten OCR model (TrOCR-style) |
   | Audio (dispatch call, statement, intercept) | Model 2 — Audio Transcription (Whisper) |
   | CCTV still frame / video | Model 3 — CCTV Detection (YOLOv8) → chained to Model 4 (ANPR) if a plate is visible in-frame |
   | Vehicle/plate photograph | Model 4 — ANPR (YOLOv8 + CRNN-BiLSTM-CTC) → chained to Vahan/RTO lookup |
   | Crime-scene photograph | Model 6 — Crime Scene Forensic Object & Detail Recognition |
   | Free-text digital note / statement | Multilingual Indic NER |

4. **Multi-file, single-case batch upload.** An investigator can drop an entire evidence folder for a case in one action; each file is tracked as its own discrete ingestion event, so a partial failure (one corrupt audio file) never blocks the rest of the batch.
5. **On-device pre-check for the mobile capture path:** auto-crop, auto-enhance, and an on-device OCR/legibility preview before submission, so a bad scan is caught immediately, not after upload.

---

## **3. Stage 2 — FIR dual-path handling (image vs. digital text)**

Digitized (CAS/e-FIR) and scanned/handwritten FIRs will coexist permanently — not a transitional state. FIR digitization is roughly 91% nationally and plateauing, with real variance by state.

* **Digital-text path.** CAS/e-FIR exports already follow the fixed IIF-1 schema — use a template/schema-aware parser here, not general-purpose document AI. It is faster, more accurate, and cheaper to run at scale than treating every digital FIR as an unstructured document.
* **Image/scanned path.** Route through OCR — a layout-aware model, since IIF-1 forms have consistent field positions worth exploiting — into the same downstream entity-extraction step as the text path, so both paths converge onto one common representation before anything else happens.
* **Reconciliation step.** When a case has both a digital FIR and scanned annexures (very common — the FIR is typed, witness statements are handwritten), merge into one case record. On field conflict, flag for human review rather than silently picking one source — a wrongly auto-resolved conflict here can propagate through the entire downstream analysis.

---

## **4. Stage 3 — Extraction models (per modality)**

Run cheap, high-precision rule and regex extractors first — phone numbers, vehicle plate formats, PIN codes, IMEI patterns — and send only the *residual* free text to the heavier NER/LLM extraction step. This cascading approach is both cheaper to run at 16,000-station scale and more auditable: a regex match for a 10-digit mobile number is trivially explainable in court, a transformer's confidence score is not.

* **Multilingual NER**, fine-tuned on Indic-language base models, since FIRs and statements are frequently in regional languages, transliterated, or code-mixed with English. A model trained only on English text will silently underperform in exactly the cases that matter most, and this failure mode is invisible unless specifically tested for.
* **Face/voice/plate matching calls out to NAFIS/AFRS as external services** rather than reimplementing biometric matching — both a legal-authorization boundary (NCRB owns these systems) and an engineering efficiency (do not rebuild a solved problem).
* **Internal, non-identifying visual signals are computed separately from external identity matching.** A cross-camera person re-identification embedding and a face-crop embedding are generated internally purely as search keys — they let the system recognize "this is the same unidentified person across three camera feeds," or later re-match new footage against an old unresolved case, without ever constituting an identity claim on their own. Identity claims come only from NAFIS/AFRS or human confirmation, never from an internal embedding alone.
* **Crime-scene visual extraction.** Model 6 runs on every crime-scene photograph submitted through the Intake Console, producing a structured, investigator-facing object inventory that feeds directly into the same Object/Item node type and MO-signature embedding used by the serial-crime playbook.
* Every extraction, regardless of method, is written with a **confidence score and a pointer back to the exact source span** — page/line for text, frame/timestamp for video, byte-offset for structured records, bounding-box/region for crime-scene and CCTV images. This single requirement, applied uniformly, is what makes every later output traceable to evidence, which is the actual precondition for court usability.

### **4.1 Crime Scene Forensic Object & Detail Recognition**

A dedicated vision model that looks at crime-scene imagery and identifies what's investigatively relevant in it, not just "what objects are present" in a generic sense — kept deliberately separate from the CCTV traffic model, since crime-scene stills have a different class distribution, a different investigative purpose, and require a free-text investigative note per object rather than a bare class label.

* **What it detects and annotates**, per object: weapon type and probable caliber/class (visual only — never a ballistics claim), blood/biological evidence and its spatial distribution, tool marks and probable tool type (pry marks, cut marks, drill marks), forced-entry indicators, drug/contraband packaging signatures, IED-component signatures (routed to a legally-gated Bomb Squad review path, never auto-published), digital devices present (phones, laptops — flagged for digital forensics seizure), personal effects that could carry latent prints/DNA, vehicles at scene (cross-referenced against the ANPR model), documents at scene (routed to the FIR OCR path if legible), environmental context (lighting, weather-exposure state of evidence), and fire/accelerant indicators.
* **Investigative-perspective notes, not a generic caption.** Each detected object gets a short structured note oriented at what an investigator needs — e.g. *"Tool mark on door frame, consistent with pry-bar forced entry; recommend toolmark comparison against database of open burglary cases in this jurisdiction"* — always phrased as a recommendation with a confidence score, never a certainty. The model is deliberately never trained to produce mechanism-of-injury or cause-of-death conclusions; those sentence types are excluded from its training data at the data-curation stage, not filtered after generation.
* **Feeds forward into**: the Object/Item graph node (`matches-MO-of` edge type), the MO-similarity ANN index used by the Serial Crimes and IED/bombing playbooks, and the Stage 5 Fact Sheet's "Evidence on file" section.
* **IED-component handling.** Any detection in the `ied_component` class is automatically routed to a Bomb Squad legal-review queue and is not written to the case graph, shown on any dashboard, or fed to any downstream model until that review clears — a hard gate enforced in the serving pipeline itself, not left to a downstream consumer's discretion.

---

## **5. Stage 4 — Entity resolution & knowledge graph**

* **Tiered resolution.** Deterministic/exact match first (shared phone number, shared vehicle plate), then fuzzy/probabilistic matching for names and addresses — transliteration-aware, not plain edit distance, since "Mohd Irfan," "Mohammed Irfan," and "इरफान" are the same person — gated by a confidence threshold: above it, auto-merge; below it, queue for human confirmation. A false merge below the confidence floor is never allowed to happen automatically — it is a wrongful-implication risk, not merely a data-quality issue.
* **Aadhaar is never used as an entity-resolution key.** It is legally fraught (post-*Puttaswamy* privacy jurisprudence, Aadhaar Act purpose limitation) and a single point of failure if ever leaked or misused. An internally generated entity ID with human-verified linkage is used for high-stakes merges instead.
* **Graph database**: a property-graph model (Neo4j-class), where every edge carries a source reference, a confidence score, and a validity time window. The temporal dimension matters because criminal networks restructure after arrests — "this edge was true in 2023" is a different claim from "this edge is true now."
* **The graph is a derived, projected view over a relational system of record, never the sole store of truth.** Courts and auditors expect a relational, replayable audit trail; a graph-only architecture makes "what did the system know on date X" much harder to reconstruct on demand.
* **Node types**: Person, Phone, Vehicle, Address/Location, Financial Account, Organization/Group, Case/FIR, Event, Document/Evidence-Item, Object/Item (weapons, IED components, seized contraband, and vehicles-as-physical-objects distinct from their registration record — needed for forensic linkage and MO-signature clustering), and PhantomEntity.
* **PhantomEntity — the load-bearing addition.** When resolution is attempted against a real-world slot implied by the evidence (a caller, a payer, a registered owner, a face in frame) and fails to resolve to any known entity, the system does not drop the signal. It creates a `PhantomEntity` node carrying whatever partial attributes exist — a partial number, an IMEI, a transaction ID, an unidentified-face or re-identification embedding, a fragment of a plate. This node is a first-class citizen of the graph with its own lead type, confidence score, and a recommended next action.
* **Edge types**: `owns`, `resides-at`, `member-of`, `involved-in-event`, `appears-in`, `witnessed`, `planted`, `purchased-from`, `paid`, `matches-MO-of`, `suspected-link-to` (a lower-confidence class, always rendered visually distinct), alongside the base set `co-accused-in`, `called`, `co-located-with`, `co-incarcerated-with`, `financially-linked-to`, `travelled-with`.

---

## **6. Stage 5 — Case Fact-Sheet Auto-Summarizer**

This stage runs immediately after entity resolution, on **every** case — Track 1 and Track 2 alike — and it runs *before* the triage gate decides the track. Both the triage decision and any human reviewer benefit from a clean, structured fact sheet regardless of how complex the case eventually turns out to be. This is the exact mechanism that identifies all the important contents of newly ingested evidence and shows them to the user: the same structured entity/event output that populates this fact sheet is precisely what Stage 7/8's GNN matching layer consumes downstream — nothing is re-extracted or re-summarized separately for the graph layer, so the UI view and the matching engine can never silently drift apart from each other.

* **Fixed, structured bullet-point output**, generated directly from the normalized entity/event schema — never a free-form summary of raw text, which would be neither auditable nor consistent across cases. The fixed sections are: **Who** (every named entity with their role), **What** (offence, statutory sections), **When** (event timeline), **Where** (locations and jurisdiction), **Evidence on file** (a list with source pointers for each item, including the crime-scene forensic object inventory), **Known relationships** (edges already resolved), and **Open gaps** (unfilled slots, forward-feeding into Stage 8's lead engine).

* Every bullet must trace to a source record; a bullet with no citation is never generated or shown.
* The fact sheet is **re-generated as a diff, never simply overwritten**, every time new evidence arrives — "what has changed since you last viewed this case" is a first-class property of this stage's output.

---

## **7. Stage 6 — Case-complexity triage gate (the answer to the open-shut case problem)**

This gate sits between the Fact-Sheet stage and the heavier graph analytics stages and is a genuinely load-bearing architectural component. It must remain rule-based and fully explainable at all times — never an opaque ML classifier, because misrouting a case has real, tangible consequences for how it is handled downstream.

**Triage signals:**

* Statutory section flags — UAPA, NDPS, ITPA/BNS-trafficking chapter, MCOCA-class state organized-crime laws, BNS unlawful-assembly/rioting sections, and repeat-MO indicators.
* Structural signals — number of distinct accused or entities above a defined threshold, presence of cross-jurisdiction elements, and financial-transaction complexity above a defined threshold.
* Explicit investigator override — an officer can always manually flag a case into Track 2 regardless of the automated signals, and can also flag a Track 2 case back down. The triage gate assists; it never overrules a human.
* **Automatic re-triage on new evidence.** If a later evidence item causes the case to cross a triage threshold it did not previously meet, the gate re-fires automatically and the case moves to Track 2 — this transition is itself logged as a first-class event on the immutable audit ledger.

**Routing outcome:**

* **Track 1 (routine).** Entity data is still written to the graph for future cross-referencing value, but no heavy analytics run, no generated "insight" is produced, and no LLM narrative is written. The output is a plain confirmation that the case has been logged, plus the Stage 5 fact sheet.
* **Track 2 (complex).** The full pipeline runs — graph analytics, the lead-chaining engine, case-type-specific playbook processing, insight generation, crime-reconstruction theory generation, and the motive/ideology hypothesis engine.

This gate is what prevents the system from manufacturing false patterns out of thin, open-shut case data. **Ideological content is never used as an automated triage or flagging trigger on its own** — Track 2 routing always requires a structural signal (statutory section, entity count, cross-jurisdiction linkage, financial complexity, or investigator override); the motive/ideology hypothesis engine described in Stage 8.5 only ever runs on cases already in Track 2 for one of these other, structural reasons.

---

## **8. Stage 7 — Graph analytics, and the GNN as the connecting layer between new evidence and historical records**

Classical, fully explainable methods — centrality measures, community detection — are the default and primary output. GNN-based link prediction is a secondary, clearly and permanently labeled "hypothesis, not evidence" layer, since it is inherently less explainable and must never be presented with the same confidence framing as a rule-derived or classical-graph-derived result.

**System design for connecting new input data to historical records via the GNN**, in order:

1. **Every new case's Stage 4 entity-resolution output is projected into the same live Neo4j graph the historical case corpus already lives in** — a new case is never analyzed in an isolated sandbox graph and then "compared" to history after the fact; it is written directly into the shared graph as new nodes/edges with their own `valid_from` timestamp and `source_ref` pointing back to the current case. This is what makes "this accused shares a phone with a 2023 case in another district" a native graph traversal, not a separate matching step bolted on afterward.
2. **The trained HGT encoder computes embeddings for every newly-written node via a single inductive forward pass over its 1–2 hop neighborhood** — no retraining. This is what lets a brand-new PhantomEntity (an unidentified caller from this morning's wiretap) sit in the same embedding space as a person node from a five-year-old closed case, and be compared to it meaningfully.
3. **Link-prediction scoring runs both "inward" (new entity against existing historical entities) and "outward" (new entity against other new entities in the same case)** — one shared scoring mechanism surfaces both "this new node is probably the same person as a 2021 case's PhantomEntity" and "these two new-case nodes are probably linked to each other."
4. **Classical graph analytics (community detection, centrality, MO-similarity ANN search) run over the same unified graph**, so a new case's accused can immediately surface as a high-centrality node in an already-known community, or immediately match an unsolved-case MO signature, without any manual cross-referencing step by the investigator.
5. **Everything produced above is written back to the case's Open Investigative Lead Board and Cross-Case Linkage output** — the connecting-layer role of the GNN surfaces directly as investigator-facing, cited, confidence-scored leads, never as a hidden backend computation.

**What is explicitly NOT a GNN task**: community detection and centrality ranking run via Neo4j GDS (Louvain/Leiden, betweenness/eigenvector/PageRank); financial-chain path tracing runs via classical Cypher path search. These stay primary and citable in court; GNN link-prediction stays secondary and visually distinct, feeding the Lead Board as a hypothesis.

---

## **9. Stage 8 — Investigative Lead & Hypothesis-Chaining Engine**

This is the component that turns the system from a lookup tool into a genuine investigation-assisting one — the direct architectural answer to "help find the next link to the perpetrators, not just check whether a record already exists." It runs continuously on every Track 2 case, driven jointly by the graph analytics of Stage 7 and the reconstruction engine of Stage 10 — the two share a single underlying gap-analysis step rather than duplicating the logic that identifies what's missing.

* **Slot detection.** The engine actively identifies unfilled relationship slots implied by the evidence on file — "financier of this act," "registered owner of this unidentified vehicle," "subscriber of the number that called the accused four times in the hour before the event," "second person visible in the frame but not yet identified."
* **Phantom entity creation.** When a slot cannot be resolved to an existing entity, a `PhantomEntity` node is created carrying whatever partial attributes are known, tagged with a lead type, a confidence score, and a concrete, legally-correct recommended next action — for example, "request CDR subscriber KYC for number X under the Telegraph Act," "request bank KYC for account Y," or "cross-check IMEI Z against other open cases."
* **Lead lifecycle, fully state-tracked.** Every phantom lead moves through: `open → data requested → data received → resolved` or `dismissed (a dead end, with a reason logged)`. Every transition is written to the immutable audit ledger. A dismissed lead is a completely normal, expected, and non-deletable outcome — "what did the system flag and why was it dropped" is itself part of the audit trail an oversight body or court may need later.
* **Multi-hop chaining, depth-limited and confidence-decaying.** Resolving one entity automatically re-triggers slot detection one hop further out — identifying the IED planter leads to pulling his priors, priors reveal a gang affiliation, that affiliation triggers a search for other members with financial or call contact to the planter in the relevant time window, and each of those becomes a new open slot in turn. This expansion is a bounded breadth-first search: each additional hop away from confirmed evidence carries a progressively lower confidence floor, and continuing the expansion past that floor requires either explicit human confirmation or new corroborating evidence — this bound is what prevents the engine from manufacturing a sprawling, unfounded network out of one shaky initial lead.

### **9.1 Stage 8.5 — Automated Motive, Ideology & Organizational-Affiliation Hypothesis Engine**

This engine generates real, cited theories on how the crime must have happened — including which organization may be involved, the perpetrator's probable ideology, how he thinks, and what patterns he follows — fully automated, with no pre-display human-review gate, running as part of the same pass that produces the Crime Reconstruction Theories in Stage 10, mechanically tied to that stage rather than a separate freeform module.

**Structural safeguards that make full automation viable, built directly into the model and the pipeline rather than into a review policy layered on top:**

* **Permitted input signals**: MO-signature match against known groups/prior cases, financial-flow pattern (funding-trail structure, payment cadence), communication-network structure (who talks to whom, timing — not message content unless already warrant-gated per the standard interception legal path), travel-pattern data (PNR/immigration where legally available), prior-case affiliation history (e-Prisons co-incarceration, prior chargesheets), and stated statements already on the case record (confessions, witness statements) where those statements exist as cited evidence.
* **Structurally forbidden input signals**: religion, caste, ethnicity, region/language of origin, or any statistical proxy for these (surname-based caste inference, PIN-code-based religious-demographic inference) are never passed to this model as a feature — excluded at the data-pipeline level, before the model ever sees them, and excluded from its training data as well, so the model has no exposure to these features in its parameter updates at all. This is a feature-engineering firewall, unit-tested in CI: a change that attempts to add a forbidden field fails and cannot merge, exactly like a regex-pattern-library change.
* **Output is a hypothesis, not a finding, on every surface it appears on** — a plain-language confidence qualifier, never a bare score, and every clause traced to a specific cited source record via the same citation-validator used everywhere else in the LLM stack. A motive/ideology sentence the validator cannot trace to a source is rejected and regenerated, never shown as-is — no exception for this output type.
* **Organizational-affiliation claims** are generated only from graph-structural evidence (shared contacts, shared incarceration, shared MO, shared financial counterparties) plus any already-confirmed intelligence inputs (IB/State SB reports already on file), and must always be jointly supported by at least one structural graph signal, never inferred from content analysis alone.
* **Mandatory standing bias-audit inclusion.** Because there is no pre-display human gate, this output type is its own first-class tracked category in the bias/fairness audit dashboard — aggregate statistics on which communities or demographics this engine's outputs disproportionately associate with organizational/ideological hypotheses are reviewed on a standing cadence, at least as frequently as the existing "key influencer" audit. This is the system's actual safety net given the automation choice: oversight moves from *before* the output is shown to *continuous, aggregate, and auditable*.
* **Investigator feedback control** applies to this output exactly as it does to every other generated insight — confirmed correct / not useful / incorrect, feeding both the retraining loop and the bias audit.

**Output fields per hypothesis**: probable motive category (financial, retaliatory, ideological, opportunistic, coerced, other — each independently cited), probable organizational affiliation with supporting structural evidence, a behavioral-pattern summary ("follows a consistent MO of X across N prior cases, operates in Y time windows, uses Z communication tradecraft"), and a recommended investigative angle derived from the above.

---

## **10. Stage 9 — Insight generation**

Retrieval-augmented generation, constrained to narrate only facts already present in the graph or in extracted evidence, with **mandatory per-sentence source citation enforced at a generation-validation step** — any sentence the validator cannot trace back to a source record is rejected and regenerated, never shown as-is. This is a hard technical requirement, not a best-effort guideline.

---

## **11. Stage 10 — Crime Reconstruction & Narrative-Theory Engine**

This stage runs alongside Stage 9, on Track 2 cases only, and builds logical, cited theories of how the crime would have exactly happened, giving the investigation a concrete direction.

* It consumes the Stage 5 fact sheet, the full evidence graph, crime-scene imagery and video (including Model 6's structured forensic object inventory), and forensic reports, and produces **several ranked theories** of how the event actually occurred. Each theory is an ordered sequence of sub-events, each sub-event tagged with its supporting evidence citations and its own confidence score.
* Each theory explicitly lists its **unresolved gaps** — "who paid the IED planter," "how did the getaway vehicle leave the cordon area." These gaps feed Stage 8's slot detection directly; theory generation and lead generation are two different presentations of one shared gap-analysis step, deliberately kept mechanically linked so they can never quietly drift apart.
* Each theory carries, where the case's playbook and evidence support it, the Stage 8.5 motive/ideology/organizational-affiliation hypothesis as a labeled sub-section — a theory and its motive hypothesis are versioned together, never generated or updated independently of each other.
* The same citation-validator enforcement used in Stage 9 applies here: a theory sentence with no traceable source is rejected and regenerated, never shown.
* **Theories are versioned, never silently overwritten.** When new evidence changes the picture, a prior theory version stays in the audit trail with an explicit pointer to what superseded it and why.
* Every theory is explicitly and permanently labeled as an investigative hypothesis, never a finding, on every UI surface that displays it.

### **11.1 Priority Lead / Suspicion-Strength Ranking**

Every candidate suspect/entity already in the case graph is scored and ranked to answer "which suspect has the highest investigative priority" — framed explicitly as an investigative-priority score, not a guilt or culprit-probability score, consistent with the system's "leads not verdicts" contract that runs through every other output.

* **Composite score, fully decomposed and shown, never a bare number**: GNN link-prediction probability to known syndicate/organization nodes + MO-similarity match strength + Stage 8.5 motive-hypothesis confidence + historical-pattern/recidivism signal (prior-case linkage strength via e-Prisons/CCTNS) + structural graph centrality within the case's own network. Each component is individually visible on click — an investigator sees exactly *why* someone ranks where they do, not just the final number.
* **Mandatory framing, enforced at the UI layer, not left to convention**: displayed as "Investigative Priority: High / Medium / Low" with the plain-language qualifier always present, and a permanent, non-dismissible label — *"This ranks investigative priority based on available evidence. It is not a determination of guilt."* — attached to the ranking wherever it is shown, including in exports.
* Feeds the Investigative Lead Board as the default sort order investigators see, and feeds the Investigative Brief only as a cited, ranked list of leads to pursue — never as a stated conclusion.

---

## **12. Stage 11 — Data governance, legal compliance, and audit (non-negotiable layer, not a bolt-on)**

* **DPDP Act 2023.** Law-enforcement processing carries certain exemptions, but purpose-limitation and audit-trail obligations remain in force. Every query against the graph is logged with a purpose and a case ID, and access is scoped to that purpose — an investigator on Case A should never be able to browse unrelated Case B's data without a logged, justified reason.
* **BSA 2023 §63 — electronic evidence certification.** Every piece of extracted or derived evidence — including Model 6's crime-scene object inventory and Stage 8.5's motive/ideology hypothesis output — needs a certificate of authenticity chaining back to its source. This is built into the pipeline as a first-class output at extraction time, not a manual afterthought.
* **Immutable, hash-chained audit log.** Every ingestion event, entity merge, query an investigator runs, insight generated, theory version, lead-lifecycle transition, motive/ideology hypothesis generation event, and Priority Lead score recomputation gets a hash appended to an append-only ledger. A permissioned ledger (Hyperledger Fabric-class) is realistic for a production government deployment; a simple cryptographic hash-chain is sufficient to demonstrate the concept at prototype stage.
* **RBAC/ABAC.** Role-based access scoped by rank, jurisdiction, and case assignment — a constable-level user sees a narrower view than a state crime-branch analyst. Redacted, clearance-tiered exports support cross-agency sharing.
* **Bias/fairness audit module.** A standing, periodic review — never a one-time pre-launch check — of which entities and communities the "key influencer" output, the Priority Lead ranking, and the Motive/Ideology Hypothesis Engine's outputs disproportionately flag or associate with organizational/ideological attribution, checked against the demographic distribution of the underlying case population, since drift happens as the model and the data change over time. Given that the Motive/Ideology Hypothesis Engine runs with no pre-display human gate, its coverage in this audit is treated as a first-class deliverable, not a footnote.

---

## **13. Stage 12 — Integration architecture for actual Indian police systems**

* **CCTNS/CAS-State** — the primary FIR/case data source, via a batch export adapter aligned to each state's own rollout maturity. Data quality and availability are never assumed to be uniform across states; the design degrades gracefully where a state's feed is thinner.
* **ICJS** — used as the intended interoperability gateway to courts, prisons, prosecution, and forensics, rather than building bespoke integrations to each of those systems individually.
* **NAFIS/AFRS** — called as external matching services, never reimplemented in-house.
* **e-Prisons** — a batch adapter for the co-incarceration network signal, a genuinely high-value and likely underused signal elsewhere.
* **NCRP (I4C)** — an adapter for cybercrime complaint data, directly relevant to the Women Safety Division mandate.
* **Vahan/Saarthi** — a lookup-on-demand adapter for vehicle and license cross-referencing.
* **FIU-IND STRs** — a case-triggered request path, never a bulk feed, since STR access is need-based and legally gated.
* **Telecom CDR/tower-dump/interception** — always case-specific, warrant-triggered pulls, never a standing bulk feed. Architecturally modeled as an on-demand adapter invoked per case, not a background ingestion source.

These integrations remain adapter/lookup-based and are deliberately not folded into the Unified Evidence Intake Console described in Stage 1.5.

**Phased rollout.** A real deployment, not a big-bang one: pilot in one or two states with mature CCTNS adoption and an active Women Safety Division use case, validate entity-resolution accuracy and investigator trust, and only then consider cross-state or additional-source-system expansion. This mirrors how CCTNS itself was rolled out over years, not switched on nationally overnight.

---

## **14. Stage 13 — MLOps and long-term maintenance (the part that determines whether this survives past year one)**

* **13.1 — Event-driven delta ingestion.** Every new evidence item is treated as its own ingestion event, scoped to that item alone. The delta pipeline re-runs only what the new item actually affects: extraction on the new item, entity resolution scoped to the newly-extracted entities against the existing graph (an incremental operation, never a full graph recompute), re-evaluation of only the case-type playbook rules relevant to that new data type, a re-check of the triage gate (including a possible Track 1 to Track 2 escalation), a re-generated diff of the Stage 5 fact sheet, a re-evaluation of open leads in Stage 8, a re-computation of the Priority Lead ranking, and — where warranted — a new, separately versioned theory and motive hypothesis from Stage 10, never overwriting prior versions.
* **13.2 — Inductive graph embeddings, a hard constraint, not an optimization.** The GNN encoder must be able to compute an embedding for a brand-new node without a full model retrain. This is what makes 13.1 computationally realistic at 16,000-station scale.
* **Investigator feedback loop.** Every generated insight, theory, lead, motive/ideology hypothesis, and Priority Lead ranking carries a mark-correct/mark-incorrect control. This feedback retrains and adjusts confidence-scoring and feeds the bias audit — it is never merely logged and ignored.
* **Model versioning and drift monitoring.** Extraction and matching accuracy is tracked over time, per model, per state and per language — a name-matching model tuned on one state's naming conventions may quietly degrade when rolled out to a linguistically different state, and this must be monitored explicitly, never assumed away.
* **A dedicated maintenance team commitment**, stated explicitly as part of the deployment plan itself.

---

# **DOCUMENT 2 — Input/Output Specification**

## **1. Design premise**

Not every case should produce a "network." A single-accused daylight theft, caught on one CCTV camera, with a confession, has no hidden network to surface — forcing the system to generate an "insight" here produces noise, not intelligence, and erodes investigator trust the very first time they see it. This document, together with Document 1, is built around the **case-complexity triage**, which routes every case into one of two tracks.

* **Track 1 — Routine/open-shut cases.** Logged into the graph for future cross-referencing value — a "closed" theft case's accused may resurface as a node in a trafficking case two years later — but no heavy network analysis is triggered and no forced insight is generated. The output here is closer to a fully solved CCTNS record: a contribution to future search, not an analysis.
* **Track 2 — Complex/network-relevant cases.** The full pipeline runs — entity resolution, graph analytics, the lead-chaining engine, case-type-specific playbook processing, insight generation, crime-reconstruction theory generation, and motive/ideology hypothesis generation. Cases are triaged in based on statutory sections, the number of accused or entities involved, cross-jurisdiction indicators, or an explicit case-type flag from an investigator.

**Track 2 is explicitly optimized for**: terrorism (UAPA), organized crime and gangs (MCOCA-type state laws), human trafficking (ITPA plus the BNS trafficking chapter), narcotics networks (NDPS), serial crimes (repeat MO across unsolved cases), communal riots and mob violence (BNS unlawful-assembly and rioting sections), cybercrime-enabled crimes against women, and kidnapping-for-ransom. Each has a genuinely different shape of "network" and needs different processing, handled through the Playbook Abstraction Layer defined in Section 4 below, so that adding a new crime category is a configuration exercise, not a redesign.

---

## **2. Complete input taxonomy**

### **2.1 Structured, born-digital records**

| Source | What it contains | Realistic access path |
| :---- | :---- | :---- |
| CCTNS/CAS FIR (IIF-1) | Complainant, accused, offence section, date/time, location, property | State CAS-State export/API, or the ICJS gateway |
| Chargesheet metadata | Formal accusation, evidence list, witness list | CCTNS or court filing systems |
| Criminal antecedents DB | Prior arrests and convictions per person | CCTNS national database |
| CDR (Call Detail Records) | Caller/callee number, IMEI/IMSI, cell tower ID plus lat/long, timestamp, duration, call type | Telecom provider, via lawful requisition (court/DM order under the Telegraph Act) |
| SMS Detail Records | Similar to CDR, message metadata only, never content | Same access path as CDR |
| Tower dump data | All device IDs connected to a given tower in a time window | Telecom provider, lawful requisition |
| Financial transaction records / STRs | Bank/UPI transaction logs, Suspicious Transaction Reports | Bank KYC (authorized request), FIU-IND under PMLA |
| Vehicle registration (Vahan) | Owner, registration, vehicle type/colour | MoRTH Vahan database |
| Driving license (Saarthi) | Identity cross-reference | MoRTH Saarthi database |
| e-Prisons records | Incarceration history, co-inmates, parole status | National e-Prisons system |
| NCRP cybercrime complaints | Cyber-harassment, financial fraud, blackmail complaints | National Cyber Crime Reporting Portal (I4C) |
| Court case status/judgments | Case progression, verdicts | ICJS/e-Courts |
| Passport/immigration (FRRO) | Foreign national tracking, travel history | Passport Seva, FRRO |
| Railway/airline PNR data | Travel movement | Requires lawful requisition |
| TrackChild / missing person DB | Missing children records | Ministry of Women & Child Development database |

### **2.2 Unstructured text**

Witness/victim statements, confession and interrogation transcripts (161/164 CrPC-equivalent BNSS statements), case diary and general diary narrative entries, chargesheet narrative sections, Intelligence Bureau/State Special Branch reports, full court judgment text, and open-source social media content — the last only ever under a warrant/legal-process framework, flagged as a hard legal gate, never a free-scrape source.

### **2.3 Images**

CCTV still frames, crime-scene photographs, evidence photographs (weapons, contraband, documents), suspect photographs, mugshots, sketch-artist composites, scanned FIRs and ID documents, and drone/satellite imagery (valuable for riot cases and large trafficking-transit-point surveillance).

### **2.4 Video**

CCTV continuous footage (fragmented ownership across private, municipal, and state cameras is a real integration headache), body-worn camera footage, and drone footage for riots and large gatherings.

### **2.5 Audio**

Lawfully intercepted call recordings (a separate legal gate from CDR — Telegraph Act §5(2)/IT Act §69 authorization), recorded witness/suspect statements, control-room/dispatch recordings, and seized-device voice notes via digital forensics extraction, warrant-gated.

### **2.6 Biometric**

Fingerprints (NAFIS, called as a service), face (AFRS, called as a service), and DNA profiles (state forensic labs, high value for serial-crime linkage across cases with no other common thread).

### **2.7 Digital forensics (seized devices)**

Mobile extraction data (chats, media, app data, location history, via authorized forensic tools), laptop/hard-drive extraction, and social media account data obtained via court order to the platform, never via scraping.

### **2.8 Cross-agency / cross-border**

Interpol notices for trafficking or organized crime with cross-border legs, and FIU-IND STRs for the financial-network angle.

---

## **3. Common schema — what every input resolves into**

Everything above must resolve into the **same underlying entity/event schema**, regardless of source or modality. This is the single most important design decision in the whole system.

**Node types**: Person, Phone, Vehicle, Address/Location, Financial Account, Organization/Group, Case/FIR, Event, Document/Evidence-Item, Object/Item, and PhantomEntity.

**Edge types**: `co-accused-in`, `called`, `co-located-with`, `co-incarcerated-with`, `financially-linked-to`, `travelled-with`, `owns`, `resides-at`, `member-of`, `involved-in-event`, `appears-in`, `witnessed`, `planted`, `purchased-from`, `paid`, `matches-MO-of`, and `suspected-link-to` — this last type is always the lower-confidence class, rendered visually distinct wherever edges are shown.

Every edge carries a source reference, an extraction confidence score, and a timestamp/validity window.

| Input type | Extraction target |
| :---- | :---- |
| FIR/chargesheet text | Person/Address/Vehicle entities, offence section, event (what/when/where), relations |
| CDR | Call-graph edges (Phone ↔ Phone), co-location edges (Phone ↔ Location ↔ time) |
| Financial records | Account ↔ Account transaction edges, amount, timestamp, structuring-pattern flags |
| CCTV image/video | Person/Vehicle detections, face embeddings, plate numbers, timestamped location |
| Crime-scene photograph | Weapon/tool/biological/digital-device Object nodes, `matches-MO-of` edges, investigative annotations with confidence and bounding-box source pointer |
| Audio | Transcribed text (routed to the same NLP extraction as text), speaker identity via voiceprint match, diarized segments |
| e-Prisons | Person ↔ Person `co-incarcerated-with` edges |
| Biometric | Person identity confirmation/deduplication signal, never a standalone entity in its own right |

---

## **4. Case-Type Playbook Abstraction Layer**

Generic "find the network" analysis is weak. Real value comes from tailoring the analysis to what each case type actually looks like on the ground — but hardcoding that per case type does not scale, since the next category the system needs to handle shouldn't require touching the pipeline. Case types are therefore formalized as **playbooks**: declarative configuration, not new code paths.

A playbook consists of:

1. **Trigger signals** — the same statutory-section and structural signals used by the Stage 6 triage gate.
2. **Relevant node/edge subset** — which parts of the schema this case type weighs most heavily. Trafficking weighs `travelled-with` and `financially-linked-to` most heavily; riots weigh `co-located-with` within a specific time window plus video-derived participation signals.
3. **Expansion-rule priority order** — the specific sequence Stage 8's multi-hop lead-chasing engine follows for this case type.
4. **Output emphasis** — which output types this playbook is expected to populate most heavily; others are allowed to stay empty.
5. **Sensitive-content gates** — for example, terrorism's rule that ideological content is never an automated flagging trigger on its own becomes a playbook-level flag, not a special case buried somewhere in the core pipeline. This gate governs *triage/flagging*; it is distinct from the Stage 8.5 Motive/Ideology Hypothesis Engine's *output generation*, which runs automatically once a case is already in Track 2 for a structural reason, subject to the input firewall and bias-audit obligations described in Stage 8.5.

**Playbook library** — designed to be extensible; a new row here is a configuration change, not a redesign:

| Case type | Expansion-rule priority order | Output emphasis | Sensitive-content gate |
| :---- | :---- | :---- | :---- |
| Terrorism (UAPA) | Financial network (funding trail) → communication network → cross-agency intelligence fusion (IB/State SB inputs) → travel pattern (PNR/immigration) | Cited investigative brief, layered financial/comms/travel overlay views | Any content-based ideological analysis is investigator-reviewed only, never an automated flagging trigger |
| Human trafficking | Route/chain reconstruction (recruiter → transporter → harborer → exploiter) → cross-border/PNR/FRRO data → financial mule detection → TrackChild integration for victim identification | Route/chain reconstruction, cross-state linkage | Victim-identity protection overrides every export and sharing rule |
| Serial crimes | MO-similarity clustering against the unsolved-case database first, network chase only where it surfaces genuinely shared entities | MO-similarity matches, geospatial pattern map | None beyond the standard governance layer |
| Communal riots/mob violence | Crowd/video analysis for instigator identification → social-media rumor-propagation network analysis (strictly legal-process gated) → dispersal and participant network mapping, kept structurally separate from any "criminal gang" network framing | Crowd/video timeline, instigator identification | Social-media analysis requires a warrant; it is never auto-scraped |
| Cybercrime against women | Repeat-offender tracking across NCRP complaints → linking pseudonymous online identities to real accounts via platform-disclosed data (court order only) → stalking-pattern detection across multiple FIRs/complaints | Cross-complaint linkage, repeat-offender flags | Platform data only via a court order, never via scraping |
| Drug cartel/narcotics network | Seizure → financial mule chain traced backward → known-route MO matching → co-incarceration network for supplier identification | Financial chain reconstruction, route reconstruction | None beyond the standard governance layer |
| IED/bombing | Communication and financial network chase around the identified or phantom planter → device-signature MO clustering against the unsolved-case index | Crime Theory Board, Investigative Lead Board, MO-similarity matches | Ideological content is investigator-reviewed only, same as terrorism |
| Serial bombing | Device-signature/MO clustering across the unsolved-case index first, network chase second | MO-similarity matches, cross-case linkage | Same as IED/bombing |
| Kidnapping-for-ransom | Financial and communication chase around the ransom-contact number as the primary rule | Investigative Lead Board (ransom-contact resolution), timeline reconstruction | Victim safety overrides speed at every step; human-in-the-loop confirmation is mandatory before any action is taken on a live lead |

---

## **5. Complete output taxonomy**

1. **Entity-relationship knowledge graph** for the case, with every edge traceable to a source record.
2. **Ranked "key influencer"/broker list**, each entry carrying the specific graph metric and underlying path that produced the ranking — never a bare score.
3. **Suspicious pattern alerts** — co-location before an incident, burner-phone churn, structuring transactions, MO repetition — confidence-scored, never auto-actioned.
4. **Cross-case / cross-jurisdiction linkage suggestions** — the single highest-value output given India's actual data-silo problem.
5. **MO-similarity matches against unsolved cases** — serial-crime specific.
6. **Route/chain reconstruction** — trafficking specific — recruiter → transporter → harborer → exploiter, with supporting evidence per link.
7. **Network disruption simulation** — the predicted structural impact of arresting a given node, used to prioritize limited resources.
8. **Cited investigative brief** — a natural-language summary in which every sentence links back to a source document or record.
9. **Courtroom-ready evidence export** — a BSA-2023-compliant certificate plus the full chain-of-custody log.
10. **Timeline reconstruction** across all modalities for a case.
11. **Geospatial pattern maps** — hotspots, movement routes, riot dispersal maps.
12. **Witness-exposure risk flag** — warns if a witness's linked identity is discoverable via the graph itself by network members.
13. **Confidence score on every single output above** — this system produces leads for a human investigator, never a verdict, and the interface must make this unmissable.
14. **Structured Case Fact Sheet** — the fixed bullet-point summary, always populated, on both tracks.
15. **Ranked Crime Reconstruction Theories** — each theory carrying a per-theory confidence score and cited sub-events.
16. **Open Investigative Lead Board** — the live list of phantom nodes and unresolved slots, each with a recommended next action. Arguably the single most operationally useful output in the entire system.
17. **Lead Resolution Audit Trail** — the full lifecycle history of every phantom node, for both investigator use and oversight-body review.
18. **Motive / Ideology / Organizational-Affiliation Hypothesis** — probable motive category, probable organizational affiliation, behavioral-pattern summary, and recommended investigative angle, always attached to a specific theory version, always cited, always labeled a hypothesis, always covered by the standing bias audit.
19. **Priority Lead / Suspicion-Strength Ranking** — a fully decomposed, non-verdict investigative-priority score for every candidate entity in a case, always shown with its five contributing components and its mandatory "not a determination of guilt" label.

---

## **6. What "real-world insight" means differently for open-shut vs. complex cases**

For Track 1 cases, the honest output is: *"No network signal detected — logged for future cross-reference."* This is stated plainly in the interface, never papered over with a manufactured pattern. For Track 2 cases, insight quality is measured by how many of the nineteen output types above are actually populated with non-trivial content — a terrorism case might populate its financial, communication, and travel outputs heavily while producing nothing on MO-similarity, since that output type is simply not relevant to that case type, while a serial-crime case is the reverse. The output layer is built to gracefully show only what is genuinely present for a given case, never to expect all nineteen outputs to be full every time.

---

# **DOCUMENT 3 — User Interface Design**

## **1. Personas**

Design for all of the following, never for one generic "user." Building a single interface that serves all of them badly is worse than building distinct, role-scoped views that each do one job well.

| Persona | Context | Primary need |
| :---- | :---- | :---- |
| Station-level officer (constable/SI) | Time-constrained, variable tech literacy, often at a low-connectivity station, the first point of evidence capture | Fast, near-zero-friction case/evidence intake, not analysis |
| District/state crime-branch investigator | The actual power user of the network-analysis features | Deep graph exploration, cross-case linkage, hypothesis testing |
| Senior officer (SP/SSP/IG) | Oversight and resource-allocation decisions | High-level dashboards, network-disruption "what to prioritize" views |
| Prosecutor/legal officer | Needs court-usable output, not raw graph exploration | Cited investigative brief, courtroom evidence export |
| Oversight/audit body (state or NCRB Women Safety Division) | Misuse prevention, compliance monitoring | Immutable audit trail, query logs, bias-review dashboard |

---

## **2. Core design principles**

* **Vernacular-first, not vernacular-added.** Regional-language UI — starting with Hindi and the pilot state's primary language — is a first-class design target from the start, never a translation layer bolted on later. Icon-heavy navigation, minimal reliance on English idiom, and voice input/output options run throughout.
* **Progressive disclosure.** The default view is simple; graph complexity, confidence-score breakdowns, and GNN-hypothesis layers are opt-in drill-downs, never the first thing a station-level officer sees.
* **Explainability is not optional decoration — it is load-bearing.** Every insight, ranking, or alert carries a visible, one-tap "why is this shown" path back to source evidence.
* **The system's confidence must never look like certainty.** Every AI-derived output carries a visible confidence indicator — never just a number, always a plain-language qualifier: "strong evidence," "possible lead," "unconfirmed hypothesis." The Motive/Ideology/Organizational-Affiliation hypothesis and the Priority Lead ranking are held to this standard at least as strictly as every other output, with a permanent, non-dismissible "not a determination of guilt/belief" label attached wherever they appear, including in exports — given that these are generated with no pre-display human review, this labeling is treated as a launch-blocking requirement, not a style preference.
* **Offline-tolerant.** Every screen a station-level officer touches works with intermittent connectivity: local queueing, a visible sync status, and no blocking spinners on core capture actions.
* **Designed differently for open-shut vs. complex cases.** A Track 1 case shows a simple confirmation screen — "logged, no further analysis triggered" — never a manufactured graph or a forced insight. A Track 2 case unlocks the full analytical interface. An investigator always knows which track a case is on, and can escalate a Track 1 case into Track 2 manually.

---

## **3. Station-level officer screens (mobile-first)**

**3.1 Case/Evidence Intake** — the mobile front-end of the Unified Evidence Intake Console.

* A camera-capture button for scanning a paper FIR or evidence photo, with auto-crop, auto-enhance, and an on-device OCR preview so the officer can confirm legibility before submitting.
* A voice-to-text option for narrative fields, in the officer's regional language.
* Auto-fill of IIF-1-standard fields when the officer is entering a digital FIR directly.
* A clear "queued for sync" indicator when offline, with auto-sync on reconnection and no re-entry ever required.
* No graph, no analytics, and no jargon on this screen at all — this persona's job ends at clean evidence capture.

**3.2 Case Status (post-submission)**

* A simple status: "Filed," "Under investigation," "Logged (routine)," or "Flagged for network analysis" — plain language, no exposed internals.
* If a case is later escalated to Track 2 by an investigator upstream, this officer sees only a notification, never the analysis itself — role-scoped access, strictly enforced.

---

## **4. District/state investigator screens (desktop-primary, tablet-capable)**

**4.1 Case Dashboard (home)**

* A list of assigned Track 2 cases, each showing its case-type tag (terrorism/trafficking/serial-crime/riot/organized-crime/cyber/narcotics/kidnapping), the triage reason it was routed to Track 2, and a one-line "what's new" summary.
* A cross-case linkage alert inbox, front and center, since this is the single highest-value, system-*pushed* output and should never be buried behind a query the investigator has to think to run.

**4.2 Graph Explorer**

* A visual entity-link canvas — person, phone, vehicle, address, account, and organization nodes — pan/zoom/filterable by entity type, edge type, or time window.
* Every node and edge is clickable to reveal its source document, extraction confidence, and, where merged, the entity-resolution reasoning behind it.
* A temporal slider to view the network as it existed at a chosen point in time.
* A toggle to show or hide GNN-predicted "hypothesis" edges, always visually distinct — dashed lines, a different colour — from evidence-backed edges, and never rendered identically to confirmed edges under any circumstance.

**4.3 Natural-language query bar**

* Free-text query in the investigator's language of choice — "phones that contacted this number in the last 30 days," "people who share a prison record with the accused" — translated into graph queries, always showing the underlying query logic on request for transparency.

**4.4 Investigative Brief view**

* The cited narrative summary — every sentence is a clickable link to its source record; a sentence with no valid citation is never shown, enforced upstream in the generation pipeline.
* A one-click export to a formatted, BSA-2023-aligned document for chargesheet or court use.

**4.5 Case-type-specific dashboards**

* **Trafficking**: a route/chain visualization with a map overlay, cross-border/PNR data surfaced distinctly.
* **Serial crime**: an MO-similarity match list against the unsolved-case index, ranked by similarity score, each with a side-by-side comparison of the matching features.
* **Riot**: a crowd/video timeline with flagged instigation moments, with the participant network kept separate from any "criminal gang" graph framing.
* **Terrorism/organized crime**: financial, communication, and travel network views presented as separate, overlay-able layers on one graph.
* **Drug cartel/narcotics**: a financial-chain trace running backward from a seizure, with route-reconstruction and co-incarceration-network overlays.
* **Kidnapping-for-ransom**: a live, time-critical view centred on the ransom-contact lead, always paired with a mandatory human-confirmation checkpoint before any recommended action is surfaced as actionable.

**4.6 Network Disruption Simulator**

* Select a node (a person), and see a simulated "before/after" view of network connectivity if that person is arrested — helps prioritize limited resources, explicitly framed as a planning aid, never a prediction of guilt.

**4.7 Feedback controls**

* On every generated insight, theory, lead, motive/ideology hypothesis, or Priority Lead ranking: a simple "confirmed correct / not useful / incorrect" control with an optional one-line reason, feeding both the retraining loop and the bias-audit dashboard. Kept lightweight, one tap, so it actually gets used rather than treated as extra paperwork.

**4.8 Investigative Lead Board**

* A Kanban-style view — open, data requested, resolved, dismissed — of every phantom-node lead attached to the case. Each card shows the lead type, its confidence score, its recommended next action, and one-tap controls to request data, mark resolved, or dismiss with a logged reason. The board's default sort order is the Priority Lead / Suspicion-Strength score, with each card's component breakdown (GNN link-probability, MO-similarity, motive-hypothesis confidence, historical-pattern signal, graph centrality) visible on hover/tap, so the ranking is never a black box.

**4.9 Crime Theory Board**

* Ranked theory cards, each expandable into its full cited sub-event timeline. Every theory's unresolved gaps are visibly linked to the corresponding card on the Lead Board. A dedicated, collapsed-by-default panel within each theory card shows the Motive/Ideology/Organizational-Affiliation hypothesis attached to that theory version: probable motive category, probable organizational affiliation, and behavioral-pattern summary, each line individually cited and expandable to its source evidence, with the standing "not a determination of guilt/belief" label pinned at the top of the panel.

**4.10 Case Fact Sheet view**

* The bullet summary, with a persistent "what has changed since you last viewed this case" diff indicator. This is the first thing an investigator should see on re-opening a case that has received new evidence.

**4.11 Unified Evidence Intake Console (desktop view)**

* A single drop-zone panel on the case workspace where any unstructured file can be added mid-investigation; shows the auto-routing decision, live extraction progress per file, and immediately surfaces new Fact Sheet diff lines and new Lead Board cards as extraction completes.

---

## **5. Senior officer screens (SP/SSP/IG)**

* A high-level dashboard: case-type distribution across the jurisdiction, cross-jurisdiction linkage counts, and a resource-prioritization view showing which currently-open cases have the highest network-disruption potential per the simulator.
* No raw graph-editing capability at this level. This persona consumes summaries and prioritization views; it does not perform entity-level analysis.

---

## **6. Prosecutor/legal officer screens**

* Read-only access to the Investigative Brief and the courtroom evidence export, for assigned cases only.
* Every claim is traceable to its source, and the chain-of-custody certificate is viewable inline.
* No graph-exploration UI is needed here — this persona wants the finished, cited document, not the analysis tool itself.
* The Motive/Ideology/Organizational-Affiliation hypothesis, being explicitly a non-evidentiary investigative hypothesis, is included in any export only in its cited, hedged form — never presented as an established fact in any courtroom-ready export.

---

## **7. Oversight/audit body screens**

* An immutable query-log viewer: who accessed what case data, when, and under what stated purpose.
* A bias/fairness review dashboard: aggregate statistics on which communities or demographics are disproportionately represented in "key influencer" flags and Priority Lead rankings, reviewed on a standing cadence, never just pre-launch — with a dedicated, disaggregated report specifically for the Motive/Ideology/Organizational-Affiliation Hypothesis Engine's outputs, reviewed with at least the same frequency as the key-influencer report.
* Misuse-alert flags — for example, an investigator querying entities outside their assigned case without a logged justification.
* Lead-lifecycle transitions and theory version history surfaced here as first-class auditable event types, on the same footing as entity merges and queries.

---

## **8. Cross-cutting UI patterns**

* **Confidence badges everywhere.** A consistent visual language — colour plus a plain-language label, never just a percentage — applied identically across every screen, so "this is a hypothesis" versus "this is evidence-backed" is instantly recognizable no matter which view an investigator is in. This treatment extends explicitly to `PhantomEntity` nodes and `suspected-link-to` edges, which must never carry the same visual weight as a confirmed entity or edge.
* **Redacted/clearance-tiered sharing.** Any exported chart or brief can be generated in a redacted form appropriate to the recipient's clearance level, for cross-agency sharing.
* **Consistent source-citation affordance.** The same "click to see source" interaction pattern appears everywhere a fact is shown, so investigators build one mental model rather than learning a different pattern per screen.
* **Track 1 vs. Track 2 visibility.** A persistent, unmissable indicator of which track a case is on, on every relevant screen.

---

## **9. What to explicitly avoid**

* Do not build one "do everything" screen that tries to serve every persona — this is the single most common failure mode in real government tool UIs and a large part of why adoption fails.
* Do not present GNN-hypothesis edges with the same visual weight as evidence-backed edges, under any circumstance.
* Do not require typing-heavy interaction for the station-level officer persona — this population's tech literacy and time constraints are real, and the design respects them rather than assuming desktop-analyst fluency everywhere.
* Do not hide the investigator feedback control behind a menu — if it takes more than one tap, it will not get used, and the system's learning loop dies quietly.
* Do not ever let the Motive/Ideology/Organizational-Affiliation hypothesis or the Priority Lead ranking render, export, or announce itself without its mandatory hedge label and citation trail — this is treated as a launch-blocking UI defect, not a style preference, given that this is the one output type in the system generated with no pre-display human review.