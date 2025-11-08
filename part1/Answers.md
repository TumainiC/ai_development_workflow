Problem Definition (6 points)

Problem statement:
Design and develop a real-time speech-to-speech translation system that enables natural voice communication between African languages (including unwritten or low-resource languages) and a pivot language (English or Swahili). The system should work in noisy, low-bandwidth, and dialect-rich environments and preserve meaning, tone, and prosodic cues.

Objectives:
- Provide accurate spoken translation that preserves semantics and pragmatic intent (target: human-evaluated semantic accuracy ≥ 85%).
- Maintain conversational latency suitable for natural dialogue (target: end-to-end < 500 ms on typical edge devices or < 1500 ms with cloud fallback).
- Support dialectal variation, code-switching, and unwritten languages via speech-only modeling techniques.
- Offer offline/edge operation with progressive feature availability when connectivity is limited.

Stakeholders:
- Primary: Local communities, field workers, healthcare providers, and market participants who need immediate cross-lingual understanding.
- Secondary: NGOs, linguistic researchers, language preservation groups, and government agencies.

Problem definition (6 points)

Problem statement
Develop a real-time speech-to-speech translation system that enables direct voice translation between African languages (including unwritten languages such as Rendille) and common lingua francas (English, Swahili). The system should support natural conversational flow, preserve meaning and prosody, and work in low-resource, noisy environments.

Objectives
- Provide accurate, real-time spoken translation across selected African languages and English/Swahili while preserving meaning and tone.
- Maintain low end-to-end latency suitable for conversation (target: <500 ms where feasible).
- Handle dialectal, phonetic, and speaker variability, including code-switching and mixed-language utterances.
- Support language preservation efforts by enabling collection and annotation workflows for unwritten languages.

Stakeholders
- Local communities and field workers (healthcare, markets, governance) who need immediate comprehension.
- NGOs, researchers, and language preservation groups documenting endangered languages.
- Developers, linguists, and local partners who build and validate language resources.

Key performance indicators (KPIs)
- Latency (ms): end-to-end time from input audio to output speech.
- Semantic accuracy (%): human-evaluated preservation of meaning (task-based evaluation).
- Per-language robustness: accuracy measured separately per dialect/region.
- Coverage: number of languages/dialects supported and proportion of population served.

2. Data collection & preprocessing (8 points)

Data sources
- Field recordings from bilingual speakers collected through structured interviews and natural conversations.
- Unlabeled speech corpora, radio broadcasts, and community audio for self-supervised pretraining.
- Parallel speech pairs where available (interpreted or translated recordings).

Potential biases and mitigation
- Gender imbalance: many datasets overrepresent male speakers.
	- Mitigation: targeted collection campaigns for underrepresented genders; balanced sampling in train/validation splits.
- Intonation and tone underrepresentation: tonal distinctions critical for meaning may be missed.
	- Mitigation: annotate tone/intonation samples, include tonal diagnostic sets, and use acoustic unitization that preserves tonal cues.
- Socioeconomic and domain bias: recordings may overrepresent formal or urban speech.
	- Mitigation: collect across domains (market, home, rituals), rural/urban locations, and age groups.

Preprocessing pipeline
1. Voice activity detection and segmentation to isolate utterances.
2. Noise reduction, volume normalization, and sample-rate standardization.
3. Self-supervised feature extraction (wav2vec 2.0 / HuBERT) or unitization to produce robust acoustic units for unwritten languages.
4. Alignment & quality filtering: automatic heuristics plus human checks to remove corrupted or misaligned pairs.
5. Metadata collection: speaker demographics, recording conditions, and dialect labels for stratified evaluation.

3. Model development (8 points)

Model architecture
- Recommended approach: modular cascaded or hybrid architecture combining:
	- ASR / discrete unit encoder (wav2vec 2.0-style) to convert speech to units
	- Neural MT trained on unit/phoneme sequences or text (mT5 / M2M fine-tuned)
	- TTS (FastSpeech2 + neural vocoder) for natural target speech
	- Alternatively, end-to-end speech-to-speech models (Translatotron 2-style) for unwritten languages where text is unavailable

Rationale
- Modular pipelines let teams leverage mature ASR/MT/TTS components and swap models depending on resource availability.
- End-to-end models reduce error propagation and are valuable for truly unwritten languages, but require more paired speech data.

Data split and safeguards
- Train/val/test: 70% / 15% / 15% with strict speaker separation between splits.
- Stratify splits by dialect, domain, and recording conditions to measure generalization.

Hyperparameters and training notes
- Learning rate schedules with warmup and decay; tune initial LR for fine-tuning (e.g., 1e-4–5e-4).
- Attention heads and model width dependent on compute (typical: 8–16 heads, 512–1024 hidden dims).
- Use gradient accumulation to simulate larger batch sizes when memory is limited.
- Regularization: label smoothing, dropout (0.1–0.3), and weight decay to reduce overfitting.

4. Evaluation & deployment (8 points)

Evaluation metrics
- Latency: end-to-end and component-level (ASR, MT, TTS).
- Semantic accuracy: human ratings (adequacy, fluency) and task-based success (e.g., medical instruction understood).
- Automated metrics: where text references exist, use BLEU / METEOR / BERTScore on converted transcripts; for unwritten languages prefer semantic/behavioral tests.
- Robustness: per-dialect accuracy, noise resilience, and code-switching handling.

Concept drift: detection & mitigation
- Definition: shifts in language use (new slang, pronunciation changes, domain shifts) that reduce model accuracy.
- Detection: continuous monitoring of confidence/error rates, periodic human-in-the-loop evaluation, and user feedback channels.
- Mitigation: incremental fine-tuning on newly collected labeled data, active learning to prioritize ambiguous or failing cases, and scheduled full retraining when needed.

Deployment challenges & solutions
- Primary challenge: supporting real-time translation on devices with limited compute and intermittent connectivity.
	- Solutions:
		- Model compression (distillation, pruning) and quantization (INT8) for mobile deployment.
		- Hybrid architecture: lightweight on-device models for core functionality with optional cloud fallback for higher-quality translations.
		- Adaptive quality: dynamically select smaller or larger models based on device and network conditions.
		- Offline-first design: bundle core language models and update with incremental downloads.

Validation & ethics
- Validate models with native speakers across demographics and dialects.
- Obtain informed consent for data collection and provide community benefits (e.g., language resources, local tools).
- Maintain transparency about limitations and provide user controls for reporting errors or offensive outputs.

Notes
- Prioritize early human evaluation and community involvement. For unwritten languages, invest in unitization and linguist-in-the-loop workflows to preserve meaning and prosody.
