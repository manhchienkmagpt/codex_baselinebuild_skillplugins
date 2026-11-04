# Readable training and inference checkpoints

Apply to optimized/basic only. Hardcore still has no training or checkpoint code.

## Training layout

Prefer a visible epoch loop with its batch loop directly in the training cell for a single straightforward training phase. There is no required train_epoch/train_one_epoch function or fixed function naming. sklearn uses fit directly. Keep model.train(), gradient clearing, forward/loss, backward, clipping, optimizer/scheduler steps and validation in their correct order.

Extract a helper only when it has real reuse or makes a complex phase meaningfully easier to read (for example alternating optimizers or multiple distinct training phases). Loss, decoding, collation and evaluation helpers remain useful when appropriate. Fewshot function boundaries are examples, not mandatory structure. Avoid generic training frameworks and unnecessary wrapper functions.

## Checkpoint contents and placement

Default to torch.save/joblib.dump directly in the branch where the primary validation metric improves. Do not create save_checkpoint just to wrap one serialization call. Keep the dictionary short: learned weights plus fitted tokenizer/vocabulary, label mappings, feature transformer/classifier or other learned state required by inference. Keep a small epoch/validation value only if needed for display or validation checks.

Fixed architecture/preprocessing/decoding constants, dataset provenance, split fingerprints and versions belong in the notebook and proposal; do not repeat a large configuration/provenance/RNG block inside every checkpoint. Do not save optimizer/scheduler/RNG state unless the user explicitly asks to resume training. Do not add resume machinery to ordinary inference checkpoints. Shorten the contents rather than compressing many statements onto one line.

For a model whose input preparation has no learned state, this may be sufficient:

```python
if val_score > best_score:
    best_score = val_score
    torch.save({"model_state": model.state_dict(), "epoch": epoch}, CHECKPOINT_PATH)
```

Add the actual fitted vocabulary/tokenizer or other necessary inference state when the task needs it. The metric direction, valid first checkpoint, ties and early stopping must remain correct; the example is not a universal maximize rule.

## Reload and review

Restore fitted mappings before constructing a fresh model from the notebook's declared architecture constants; derive vocabulary/output dimensions from the restored state. Reload all trained weights, including a fine-tuned backbone, without fetching pretrained weights again. Evaluate only the restored pipeline. Keep existing essential checkpoint keys where practical so same-architecture older checkpoints with extra metadata remain readable; do not silently accept incompatible weights/tokenizers.

Verify predictions before/after serialization, and compare parameter updates/loss/learning-rate order when refactoring training helpers into loops. Both inline loops and justified helpers are valid. Static validation does not establish numerical equivalence, complete inference state or resume capability. Document actual checks and missing dependencies honestly.
