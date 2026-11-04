# Focused Vietnamese solution report

Write huong_giai_quyet.md for a learner who needs to understand the problem and the optimized solution. Use plain Vietnamese, explain technical terms when introduced and connect each choice to actual data, metric and compute constraints. Architect-only requests remain report-only.

## Required structure

Use the following title and exactly these five level-2 headings, in order. Subheadings are optional; every section needs substantive content.

# Phân tích bài toán và đề xuất hướng giải quyết
## 1. Giải thích bài toán
## 2. Phân tích dữ liệu và các hướng tiếp cận
## 3. Tiền xử lý dữ liệu tối ưu
## 4. Lựa chọn mô hình tối ưu
## 5. Siêu tham số và lý do lựa chọn

1. Explain the concrete goal, raw inputs and expected outputs, one clearly labeled illustrative example and the official primary metric/direction in accessible terms. Examples are not fabricated predictions or benchmark scores.
2. Identify the real dataset/source/schema/splits and relevant verified scale, imbalance, noise, sequence/image/clip lengths or geometry. Distinguish current inspection, previously recorded observations, source documentation and assumptions. Explain leakage/grouping issues. Compare 2–4 plausible approaches using those data characteristics, metric and resource constraints; do not rank architectures by name alone.
3. Describe the selected optimized preprocessing pipeline and why each step helps this task. Include train-only fitting and correct target alignment/masks or synchronized spatial/temporal transforms. Explain important exclusions, such as not deleting negation, not changing QA character offsets or not flipping directional targets. Include only transformations the selected method actually uses; do not invent optional steps as implemented behavior.
4. Choose one primary method and explain its architecture/dataflow, why it fits better than the alternatives under the stated constraints, its costs and likely failure modes. Keep pretraining eligibility/provenance here when relevant. For existing artifacts, match baseline.ipynb rather than silently proposing a different implementation. A simpler model may be the strongest defensible choice under a scratch-only or small-data constraint.
5. Include a task-specific table: parameter | value | role | reason for this value | effect of increasing/decreasing. Use the actual optimized notebook constants, resolving aliases and noting any validation-selected candidates. Explain tradeoffs at the selected value; do not say a value is optimal merely because it is common. For sklearn, explain features/regularization/solver settings rather than inventing batch size/epochs. For neural tasks, include the relevant learning rate, batch, capacity, regularization, length/image limits, training budget and decoding settings. Fixed paths, seeds and cache flags are not a substitute for hyperparameter analysis.

## Evidence and scope

“Optimal” means justified within the dataset, primary metric, model/pretraining policy and actual compute budget. Use “đề xuất có cơ sở” for untested choices; say “tốt nhất đã đo trên validation” only with comparable validation evidence. Do not fabricate a score, class distribution, GPU fit, ablation or hyperparameter sweep. Distinguish a candidate search set from a selected value. Discuss sensitivity or a focused validation experiment when useful, without claiming it ran.

The report concentrates on the optimized method. Mention basic/hardcore briefly only if it explains a relevant alternative; no required three-variant comparison table, learner roadmap or separate basic/hardcore chapters. Keep the three-notebook generation contracts unchanged.

Do not add numbered sections for notebook layout, checkpoint implementation, Run All, runtime paths, hashes, logs, verification history or an appendix by default. State only concise experimental limitations relevant to the chosen method within the five sections. Execution/check status belongs in the final handoff or concise notebook notes, not a long report history. Retain relevant verified data observations, source links and genuinely measured comparable results; discard stale claims that cannot be tied to the current method/protocol.

## Review

Check all five sections manually for explanatory substance and consistency with the actual optimized model, preprocessing, constants, primary metric and pretraining restrictions. Static validation checks headings/order/nonempty sections only; it does not prove that data or performance claims are true. No fixed word count or generic parameter list is required.
