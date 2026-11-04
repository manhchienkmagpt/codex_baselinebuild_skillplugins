# Image and video task behavior

| Task | Basic scratch candidate | Optimized direction | Correctness / metric |
|---|---|---|---|
| Classification / image pairs | Small CNN, pooled features and simple classifier; shared scratch CNN for pairs | Eligible backbone, task-valid train augmentation and stronger fusion | Same identity/patient/group split; label IDs; accuracy or official macro-F1/AUROC; no pair leakage |
| Detection | Small scratch CNN with simple grid/anchor-free head | Stronger feature backbone, multi-scale scratch detection head | Synchronize box transforms; xyxy/xywh and pixel/normalized coordinates explicit; original-size output; official mAP and empty GT/pred handling |
| Semantic / instance segmentation | Small encoder-decoder; scratch mask/object heads for instances | Eligible backbone, scratch multi-scale/mask heads | Nearest-neighbor mask resize, ignore index, synchronized geometry; mIoU/Dice or mask AP as specified; keep instance IDs |
| Keypoints | Small scratch heatmap/regression network | Backbone and scratch spatial head | Transform coordinates/visibility together; left/right permutation on flip only in optimized; official OKS/PCK/AP |
| Restoration | Small scratch conv encoder-decoder | Stronger self-built residual architecture, eligible feature encoder if useful | Paired images aligned; split by original scene; output resolution/range/channels; official PSNR/SSIM range/crop rules |
| Video classification / action recognition | Small frame CNN plus mean temporal pooling or scratch RNN | Eligible video/image backbone with scratch temporal/head modules | Split by video/person/session; fixed explicit clip sampling; sample/aggregate all evaluation clips by official protocol, no train/eval frame leakage |
| Tracking | Scratch detector and simple explicit association | Backbone, stronger self-built detector/association | Ordered frames; reset state per sequence; stable track IDs; export original coordinates/frame IDs; official HOTA/IDF1/MOTA rather than classification accuracy |

Basic performs color conversion, resize/tensor and only required target/temporal conversion. No augmentation, optical-flow feature engineering or sophisticated preprocessing. Correct masks/geometry/association/output assembly remain part of the task. Basic training is complete and meaningful. Optimized preprocessing must be justified for actual labels: no arbitrary flips in directional tasks or transformations that change targets incorrectly.

Preserve official splits and raw sample/group membership across variants. Derive validation from train by subject/scene/video/sequence when necessary, never individual correlated frames. Fit any learned statistics on train only. Choose thresholds, NMS, schedules and checkpoints on validation only. Test may lack labels: still generate predictions; do not invent metrics or treat missing GT as zero.

Hardcore uses the same split/protocol with one untrained scratch raw-input model and direct inference. Output structures remain task-valid even when predictions are poor. Video loading/decoding may read frames, but fitted sampling/features/transforms are not supplied outside the model; preserve evaluation sequence/frame order and reset tracking state explicitly. No training or best-checkpoint step. Model output must actually depend on its learned scratch layers, not copied labels or independent random arrays.
