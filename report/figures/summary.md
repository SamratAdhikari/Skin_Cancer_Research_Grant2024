# Auto-generated results summary

## Baseline vs. multimodal (test set)
| Stage | Model | Acc | Macro-F1 | Sensitivity | Spec | ROC-AUC | McNemar p |
|---|---|---|---|---|---|---|---|
| Stage 1 | baseline | 0.802 | 0.681 | 0.478 | 0.880 | 0.818 |  |
| Stage 1 | multimodal | 0.827 | 0.720 | 0.536 | 0.898 | 0.852 | 0.010 |
| Stage 2a | baseline | 0.783 | 0.623 | — | — | 0.914 |  |
| Stage 2a | multimodal | 0.793 | 0.605 | — | — | 0.907 | 0.446 |
| Stage 2b | baseline | 0.676 | 0.567 | — | — | 0.828 |  |
| Stage 2b | multimodal | 0.642 | 0.506 | — | — | 0.771 | 0.275 |

## Diffusion ablation (malignant subtype)
| Variant | akiec recall | macro-F1 |
|---|---|---|
| no_synth | 0.184 | 0.506 |
| with_synth | 0.265 | 0.637 |

## End-to-end pipeline

- **stage1_accuracy**: 0.832
- **malignant_sensitivity**: 0.601
- **end_to_end_subtype_accuracy**: 0.682
- **n_test**: 1503