# Hardcore: load data, define one model, infer

baseline_hardcore.ipynb is independently runnable with the declared real data. It intentionally supplies no preprocessing pipeline or training implementation. It has exactly two numbered content sections; cells within those sections are flexible.

## 1. Load data

Put imports, fixed paths/seed/device and schema-aware data reading here. Loading helpers may read files or select raw splits, but must not tokenize, normalize text, engineer features, fit vocabulary or train. Preserve raw input content and task IDs. Selecting/grouping the declared split or assembling dialogue input fields is data loading, not permission to supply a preprocessing pipeline. Prefer supplied/local data, provide the actual dataset source/version and actionable access errors; do not fabricate fallback data.

Use test if available (labeled or unlabeled); otherwise use the declared validation/held-out. For a dataset with an established practice split, preserve its exact membership. Explicitly label held-out from validation and never call it official test. Do not use train as an implicit fallback.

## 2. One model and immediate inference

Define and instantiate exactly one task-appropriate scratch nn.Module, with randomly initialized learned parameters. Its forward accepts raw task inputs, not gold targets. Minimal mechanical conversion is allowed inside model methods only: fixed byte/character indexing, padding/truncation, PIL-to-tensor/resize, simple output decoding. It must not fit a tokenizer/vocabulary/scaler or load pretrained/other-variant weights. Fixed label catalogs are task schema, not learned features.

Directly call that same model in eval() and torch.inference_mode()/no_grad(), using every record of the selected split in stable order. Predictions must originate from its forward result, not random labels or copied gold. Preserve task-required output structure (e.g. valid spans/coordinates, original label IDs, text for generation); no quality guarantee is implied.

No external preprocessing/transform/tokenizer functions/classes, Dataset feature pipeline, training helpers, loss/optimizer/scheduler, backward, fit/fit_transform, epoch loop, checkpoint load/save, TODO/pass stubs or dependence on other variants. Library imports/setup may share the first section; do not add a third content section for training or exercises.

## Declaration and checks

Use metadata.cv_baseline.variant="hardcore", pretraining_policy="scratch", pretrained_components=[] and training_status="untrained". Declare inference={"split": "test", "id_field": "id", "prediction_field": "prediction", "prediction_type": "string"}; use the actual split, field names and type. Supported types are string, integer, spans, entities, classification, detection, segmentation, keypoints, restoration, video_classification and tracking. CV uses the same fields under metadata.cv_baseline.

Include runtime assertions for nonempty split, unique IDs, prediction count/order and task output schema. A static validator cannot establish real dataflow or semantic correctness; smoke-test inference and compare model parameters before/after. Scratch predictions are explicitly untrained execution examples; do not report them as trained quality or invent scores. When exercises are explicitly requested, explain them in notebook prose or the handoff; do not require them in the focused solution report.
