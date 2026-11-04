# Pretraining policy: scratch NLP and image-to-text components

Read this policy before selecting, generating or reviewing any model in the three NLP skills. The unit of permission is the **overall task**, not a renamed helper or a single notebook cell. Explicit task-specific restrictions, including competition rules forbidding external weights, still apply.

Apply it per notebook under the [dual-notebook contract](dual_notebook_contract.md). The optimized image-to-text notebook may use the eligible components below; the basic and hardcore notebooks require scratch with no pretrained components, even for image tasks. All variants keep their own truthful metadata, runtime assets and model provenance.

## Allowed scope

- **Text-only NLP:** use `scratch`. No pretrained/fine-tuned weights, embeddings, tokenizers, text encoders or decoders. PyTorch `torch.nn` architectures initialized from scratch, raw Hugging Face datasets, and tokenizers fitted on train are allowed. Do not import Hugging Face `transformers` model implementations for these tasks.
- **Image-to-text:** may use `image_components`. A pretrained model from Hugging Face or an ImageNet backbone from `torchvision` (e.g. ResNet, EfficientNet, VGG, ConvNeXt) must solve only a proper subtask. Freeze or fine-tune it based on data, validation and the task's fixed compute budget (32 GB GPU planning target); initialize new heads, connectors and decoders from scratch. Scratch-only image solutions remain allowed.
- Do not use a pretrained model that can complete the overall task. A wrapper, renaming it as an encoder, or loading a complete model and ignoring/removing its pretrained decoder does not make it eligible. Choose an encoder-only checkpoint/model instead. Replacing a vision classifier's ImageNet classification head with a new task head is allowed: ImageNet classification does not complete image-to-text.
- Hosted inference/model/embedding APIs, `torch.hub` loaders and pretrained sources outside the two allowed sources remain outside this exception. Run permitted components locally in the notebook. Inspect the actual model card, architecture and checkpoint contents; model names alone do not establish eligibility.

| Overall task | Allowed | Disallowed |
|---|---|---|
| Image captioning | Pretrained **image encoder only** plus a caption decoder and tokenizer/vocabulary trained from scratch | Pretrained text encoders, language decoders, caption tokenizers, or a complete pretrained captioning model |
| Standalone OCR | Pretrained image encoder plus a recognizer/CTC head or text decoder trained from scratch; train-only character vocabulary | A pretrained OCR model that already maps the input image to the requested transcript, such as a complete pretrained image-to-text encoder-decoder |
| Table extraction from images | Separate pretrained table-structure detector and cell-OCR model; implement cell assignment, text placement, ordering, merged cells and final table serialization yourself | One pretrained model that directly outputs the complete target table/HTML/JSON |
| Other image-to-text tasks | Justified visual components with intermediate outputs and an independently implemented remaining pipeline | A pretrained model or hosted service that supplies the overall target output |

A cell-OCR model may include a pretrained text decoder/tokenizer because its output is only one intermediate part of **table extraction**. The same model is prohibited for a standalone OCR task it already completes. Do not split or rename a standalone task artificially to obtain this exception. For other image-to-text tasks, use an image encoder or a justified `vision_component`; a text-generating subcomponent is permitted only for the explicitly described table cell-OCR role.

## Processors, provenance and notebook responsibilities

- A fixed image processor/normalization associated with an eligible encoder is allowed. Pretrained tokenizer/processor text capabilities are permitted only for a valid cell-OCR component in table extraction. For captioning use an image-only processor and a new caption tokenizer/decoder. Any additional fitted transformations, vocabulary, labels or fine-tuning must use train only.
- Record each component's source, model ID, exact checkpoint revision/weight enum, role, input/output, freeze/fine-tune decision, and why it cannot complete the overall task. In the proposal, include this component table under the chosen method and explain the independently implemented remainder. Check license/access and known pretraining/evaluation overlap; do not assert no contamination without evidence.
- **Configuration section:** expose the policy, component IDs/revisions, cache directory, download/offline choices, and freeze/fine-tune settings in the readable fixed configuration. Explain authorized downloads and actionable access/offline errors. Do not silently substitute random weights or a different model. Mirror policy/component declarations in notebook metadata as specified below.
- **Preprocessing section:** apply the selected weights' image preprocessing while preserving task requirements such as OCR aspect ratio. Restrict bundled text processors/tokenizers to eligible cell-OCR; train new target vocabularies on train only.
- **Architecture section:** load only eligible pretrained components, then construct randomly initialized remaining layers. Track frozen/trainable parameters; for a frozen encoder, account for BatchNorm/dropout mode when training the new layers.
- **Training/evaluation sections:** choose checkpoints/thresholds on validation, save new/fine-tuned weights and fitted tokenizer/processor state needed for inference. Keep provenance and fixed decoding/assembly settings in the notebook/proposal. Save optimizer/scheduler/RNG only for an explicit resume request. Reload the same complete pipeline for held-out evaluation. Do not fine-tune or fit any stage on validation/test. Report genuine results only.
- Use Vietnamese explanations, natural Markdown/code sections, fixed constants and ordinary PyTorch precision under the [notebook contract](../../nlp-notebook-engineer/references/notebook_contract.md) and [runtime configuration](../../nlp-notebook-engineer/references/runtime_config.md). Compare feasible scratch and component-based choices where useful; permission does not imply that a larger pretrained component is the best choice.

## Notebook metadata and static validation

Store the declaration in `notebook.metadata.nlp_baseline` (not an extra execution/setup cell). Existing notebooks without it default to `scratch`. A scratch notebook may omit the declaration, or set `pretraining_policy="scratch"` and an empty component list. Set `task_type` to `text_only`, `image_captioning`, `ocr`, `table_extraction`, or `image_to_text`; use the most specific applicable type. `image_to_text` is for other image tasks, not a way to bypass captioning/OCR rules.

Example for an ImageNet encoder used by captioning; IDs below are metadata examples, not mandatory model recommendations:

```json
{
  "nlp_baseline": {
    "task_type": "image_captioning",
    "pretraining_policy": "image_components",
    "pretrained_components": [
      {
        "source": "torchvision",
        "model_id": "resnet50",
        "revision": "ResNet50_Weights.IMAGENET1K_V2",
        "role": "image_encoder",
        "input": "image",
        "output": "spatial image features",
        "training": "frozen"
      }
    ]
  }
}
```

For `image_components`, require a nonempty component list. Every component has the nonempty string fields shown above; `source` is `huggingface` or `torchvision`, and `training` is `frozen` or `finetune`. Hugging Face `model_id` is the checkpoint repository/path passed to its loader, and `revision` identifies the chosen checkpoint. A `torchvision` ID is the builder name (e.g. `resnet50`) with its concrete ImageNet weight enum in `revision`; prefer a versioned enum over `DEFAULT`, and record the resolved enum if using `DEFAULT` during exploration.

Allowed roles:

- `image_captioning`, `ocr`: `image_encoder` only.
- `table_extraction`: `image_encoder`, `table_structure`, `cell_ocr`.
- `image_to_text`: `image_encoder`, `vision_component` (explain the visual intermediate output).
- `torchvision` ImageNet weights: `image_encoder` only, for any allowed image task.

Declare each `(source, model_id)` once. A model and its associated processor share the same component entry. Use literal IDs or statically resolvable config strings in loaders where practical. Include every downloaded pretrained component; a metadata label does not authorize an incompatible API/model.

The validator keeps its existing CLI. It checks metadata, common loader/import aliases, weight selection and recognizable incompatible full-task/text-generating use. It does **not** download models, inspect remote checkpoint capabilities, prove dataflow correctness, or certify the component boundary. Every `image_components` result includes a warning requiring model-card and actual pipeline review; dynamic/unresolved loaders receive additional diagnostics. Review and document those findings, not just the exit code. An unknown architecture that passes the static scan still requires semantic review against this policy.

## Variant requirements

Declare variant optimized/basic/hardcore. Basic and hardcore require pretraining_policy="scratch" and pretrained_components=[]. Hardcore also declares training_status="untrained" and its inference contract; it has no fine-tuning, training or checkpoint steps. Detailed training/checkpoint responsibilities above apply to optimized/basic only. Existing explicitly authorized evaluation-only components are not task-model permissions and must never participate in prediction; do not add pretrained evaluators to hardcore, which only demonstrates inference.
