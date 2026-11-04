# NLP and image-to-text decision playbook

This is a **candidate-generation guide**, not a claim of SOTA. Data quality, dataset size, compute and metrics outrank architecture fashion. Apply the [shared pretraining policy](../../nlp-baseline-workflow/references/pretraining_policy.md): scratch-only text NLP, optionally eligible pretrained components for image-to-text.

The candidates below primarily guide the optimized reference. Also select a materially simpler scratch practice method under the [dual-notebook contract](../../nlp-baseline-workflow/references/dual_notebook_contract.md), such as vanilla RNN context, a custom small CNN, limited word-only features or greedy decoding. Preserve a correct full pipeline and common evaluation; relative performance remains empirical, not guaranteed by an architecture's name.

| Task | Policy-compliant candidates | Main metric / traps | High-impact decisions |
|---|---|---|---|
| Text/topic/sentiment/spam/intent classification | TF-IDF word+character n-grams with LinearSVC/LogisticRegression for small datasets; trainable embedding+BiGRU/CNN; TransformerEncoder from scratch if sufficiently large labeled data | Macro-F1 for imbalance, Accuracy for balanced classification, task official metric | Word+char features for morphology/OOV, stratified validation, loss weights, decision threshold on val for binary F1 |
| Token labeling: NER / POS / chunking | CRF with lexical/character features, BiLSTM-CRF with learned embeddings, scratch TransformerEncoder with structured decoding | Entity-span micro-F1 for NER, token Accuracy for POS, exclude padding and non-target tokens | Handle BIO legality, align subword labels, char features, class imbalance and label transition constraints |
| Word segmentation | Character-level BiLSTM-CRF or TransformerEncoder with BMES tags; deterministic lexicon baseline | Word F1 | Corpus-specific segmentation guidelines, legal transitions, exact word boundary reconstruction |
| Machine translation | Separate source/target BPE trained on train only; scratch encoder-decoder Transformer (`nn.Transformer`), or attention-based GRU when very low resource | SacreBLEU (specify tokenizer/signature), chrF | Length bucketing, padding/casual masks, label smoothing, warmup schedule, beam search with length penalty tuned on val |
| Abstractive summarization | Scratch encoder-decoder Transformer only with enough paired examples; for very limited data test extraction/ranking as a credible alternative | ROUGE-1/2/L with documented normalization | Input truncation strategy, copy/rare token issues, beam/min-length constraints, factual consistency |
| Sentence similarity / paraphrase | Word+char TF-IDF with cosine/linear classifier; scratch Siamese encoder trained on labeled pairs, contrastive objective if feasible | Spearman/Pearson for STS, F1 for binary paraphrase | Hard negatives without leakage, calibration, length normalization |
| Retrieval / ranking | BM25 strong unsupervised baseline; sparse supervised ranker; scratch dual-encoder or cross-encoder where enough labeled data | MRR@k, nDCG@k, Recall@k (match official metric) | Candidate retrieval coverage, negative sampling, deduplication, no contamination between queries |
| Extractive QA | Trainable bi-encoder/TransformerEncoder + start/end span heads from scratch | Exact Match / token-level F1 | Windowing long passages, valid start/end pairs, answer span alignment, no-answer threshold on val |
| Language modeling | Scratch causal Transformer / RNN-LM | Perplexity (correct token masking) | Train-only tokenizer, context window, causal masking, tied embeddings if appropriate |
| OCR / image-to-text (language-related multimodal) | Scratch CNN+BiLSTM+CTC; or eligible pretrained image encoder + scratch sequence model/CTC head/decoder | CER, WER, sequence accuracy | No complete pretrained OCR model for standalone OCR; aspect ratio, CTC lengths, train-only character vocabulary |
| Table extraction from images | Separate pretrained table-structure detector and cell-OCR model, or scratch components; independently implement cell assignment and final table assembly | Task-defined structure F1 / TEDS / cell accuracy | Cell OCR may include its pretrained tokenizer/decoder; no single pretrained full-table solution; merged cells and reading order |
| Image captioning | Scratch image encoder, or eligible pretrained image-only encoder (HF or torchvision ImageNet) + scratch caption decoder/tokenizer | CIDEr/BLEU (match official metric) | No pretrained text encoder, language decoder or complete captioning model; image preprocessing, train-only target vocabulary |
| Other language-related visual tasks | Scratch model or justified pretrained visual subcomponents under the shared policy | Task-specific official metric | Verify intermediate output and independently implemented remainder; do not reclassify captioning/OCR to bypass their rules |

Use the [fewshot guide](../../nlp-notebook-engineer/references/fewshot_guide.md) for notebook organization, not a universal model choice. Keep fixed task-specific hyperparameters and ordinary PyTorch default precision; do not add hardware-adaptive configuration.

## Method-choice reasoning

- **Data scarcity:** TF-IDF+linear models, CRF, RNN inductive bias and careful features may deliver better validation scores than data-hungry scratch Transformers. Do not default to Transformers when training data is too small.
- **Large, high-quality corpora:** scratch Transformers become attractive with adequate token count, optimization steps and GPU memory. Use PyTorch attention modules, not pretrained Hugging Face classes.
- **Pretraining boundary:** text-only NLP rejects pretrained weights/tokenizers/embeddings. For image-to-text compare eligible visual subcomponents (and table cell-OCR) against scratch alternatives; inspect model cards and do not choose pretrained full-task solutions. Hosted model services remain disallowed.
- **Tuning priorities:** correct split/metric → data integrity/leakage → representation/tokenization → architecture capacity → training schedule/balancing → decoding/threshold. Keep a held-out test pristine.
- **Empirical claims:** only report scores actually obtained from training/evaluation runs, including seeds and validation/test provenance. All other statements are hypotheses.

## Basic and hardcore adaptation

Optimized candidates above retain their task-specific preprocessing. Basic uses minimal word/character vocabulary/padding or unigram counts, no BPE/advanced features/augmentation, while keeping the simpler model and complete correct training. Hardcore uses one untrained scratch model accepting raw inputs and immediate inference only; no external preprocessing or training. Follow the [hardcore contract](../../nlp-baseline-workflow/references/hardcore_contract.md).
