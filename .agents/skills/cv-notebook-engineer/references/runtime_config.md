# Fixed runtime configuration for fewshot-style notebooks

Plan for one NVIDIA CUDA GPU with 32 GB VRAM in Jupyter or a suitable cloud/Colab runtime. This is a method-design target, not an enforced device minimum, an assumed Colab allocation or a runtime memory controller. Use concise fixed constants like the fewshot examples. Users can edit them manually for other GPUs or datasets.

## Configuration style

- Prefer uppercase variables near the start over a large environment-management/config framework. Choose explicit task-specific values; the example below is illustrative, not a universal optimum or a guarantee of memory fit.
- Use ordinary PyTorch default precision. Do not generate AMP/autocast/GradScaler, BF16/FP16 switches, auto precision or a precision menu.
- Keep only simple CUDA availability/device selection and normal seed handling. Do not query total/free VRAM or compute capability, calculate memory budgets, derive batches from hardware, or catch OOM to retry with smaller settings automatically.
- Set batch/eval batch, epochs, LR, lengths/image sizes and model dimensions explicitly. Accumulation defaults to 1 and should be omitted unless the model needs a justified fixed larger value. Do not use None/auto placeholders for batch selection.
- If an actual run exceeds memory, explain which constant to reduce manually and how to rerun. During authoring a verified manual adjustment can become the new fixed default; no adaptive control logic belongs in the generated notebook.
- Keep the same raw split/metric protocol across variants; method-specific constants may differ and must match the proposal.

~~~python
from pathlib import Path
import random
import numpy as np
import torch

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/task_slug/artifacts/optimized")
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Thiết bị: {DEVICE}; batch: {BATCH_SIZE}; số epoch: {EPOCHS}")
~~~

Use the corresponding artifacts/basic directory in the basic notebook. Add concrete tokenizer/model/decoding settings when needed, not a mandatory hardware-profile section. Default data/output paths are portable and directly editable; Colab /content or user-selected Drive paths may be used when actually appropriate.

## Environment and execution

Reuse the environment's installed PyTorch and check necessary task dependencies only. A local CPU-only PyTorch installation does not enable CUDA: provide a concise setup note rather than blindly replacing wheels or drivers. Install only missing task packages if necessary; a separate setup cell is allowed. Colab Drive mounting must be optional/guarded or explicitly requested, not copied unconditionally from a reference.

Keep all selected notebooks independently runnable from raw data and declared eligible weights under the [pretraining policy](../../cv-baseline-workflow/references/pretraining_policy.md). Required parameters/model downloads are explicit; missing data/weights produce useful errors, not silent substitution.

## Verification and final cleanup

Distinguish syntax/static checks, CPU smoke tests and actual GPU execution. Do not claim a measured 32 GB fit, training time or score without running. If data permits, test forward/backward, checkpoint save/reload and the metric under the fixed configuration. Follow the [dual-notebook contract](../../cv-baseline-workflow/references/dual_notebook_contract.md): clean agent-created verification/runtime files before handoff while retaining primary notebooks/proposal. User Run All may recreate its own checkpoints normally.

## Hardcore exception

Hardcore uses fixed inference dimensions/batch/seed/device only, with no epochs, LR, optimizer, training, checkpoint save/reload or backward checks. Smoke-test direct untrained inference and unchanged parameters instead. Use artifacts/<task>/hardcore for optional prediction outputs.

## Training layout and minimal checkpoints

Use the [training style contract](training_style.md): inline loops by default for simple tasks; helpers only when useful; concise inference checkpoints saved directly on validation improvement. No mandatory train_epoch/save_checkpoint, no optimizer/scheduler/RNG serialization unless resume is explicitly requested. Fewshot helper boundaries and verbose save dictionaries are not requirements.
