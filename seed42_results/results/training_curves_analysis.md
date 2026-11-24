# Training Curves Analysis

_Auto-generated from saved training_log.csv files. All values derived from data._


## E1 Baseline (E1-R*)
- **Best val Dice**: 0.7533 at epoch 175/200
- **Final val Dice**: 0.7530 (best→final gap: +0.0003)
- **Train-val Dice gap at best epoch**: +0.0930
- **Final train Dice**: 0.8509
- **Val Dice volatility (std, last 20 epochs)**: 0.0001
- **Plateau epoch**: 101 (first epoch where val Dice ≥ best−0.005)

## E2 Boundary
- **Best val Dice**: 0.7522 at epoch 199/200
- **Final val Dice**: 0.7522 (best→final gap: +0.0000)
- **Train-val Dice gap at best epoch**: +0.0953
- **Final train Dice**: 0.8507
- **Val Dice volatility (std, last 20 epochs)**: 0.0003
- **Plateau epoch**: 134 (first epoch where val Dice ≥ best−0.005)

## E3 Consistency
- **Best val Dice**: 0.7539 at epoch 185/200
- **Final val Dice**: 0.7536 (best→final gap: +0.0003)
- **Train-val Dice gap at best epoch**: +0.0921
- **Final train Dice**: 0.8484
- **Val Dice volatility (std, last 20 epochs)**: 0.0001
- **Plateau epoch**: 121 (first epoch where val Dice ≥ best−0.005)

## E4 Combined
- **Best val Dice**: 0.7587 at epoch 184/200
- **Final val Dice**: 0.7585 (best→final gap: +0.0002)
- **Train-val Dice gap at best epoch**: +0.0889
- **Final train Dice**: 0.8498
- **Val Dice volatility (std, last 20 epochs)**: 0.0001
- **Plateau epoch**: 121 (first epoch where val Dice ≥ best−0.005)

## Interpretation
- **E1** peaks at epoch 175, well before the 200-epoch limit, suggesting the model converges cleanly under the segmentation-only objective.
- **E2** peaks at epoch 199 (last epoch), suggesting the boundary loss slows convergence or the model continues to improve; a longer training budget or early stopping on boundary metrics might be warranted.
- **E3** peaks at epoch 185; the consistency loss regularizes the model, consistent with slightly reduced train Dice but maintaining validation performance.
- **E4** peaks at epoch 184 and achieves the highest validation Dice (0.7587), suggesting the two auxiliary losses are complementary on the validation distribution.
- Train-val gaps across all models are modest (~0.09–0.10 at best epoch), indicating limited overfitting for a 0.0495M-parameter model.
- The gap between best-epoch val Dice and final-epoch val Dice is small for E1/E3/E4 but essentially zero for E2 (peaked at epoch 199), confirming model selection on EMA val Dice is appropriate.