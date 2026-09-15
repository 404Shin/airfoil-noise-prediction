# Airfoil noise prediction with PyTorch

A scientific machine-learning portfolio project by **Shayan Imran Lashari**, comparing small neural networks with simple baselines on NASA airfoil wind-tunnel measurements.

**Question:** Can a neural network predict airfoil noise better than linear regression on held-out aerodynamic condition groups, and how do regularisation and model size affect performance?

## Main result

The validation-selected network achieved **2.374 ± 0.111 dB test RMSE**, compared with **5.087 dB** for linear regression: a **53.3% reduction in RMSE** on this split. This is a reduction in a prediction-error metric, not percentage accuracy or a reduction in physical aircraft noise.

| Model | Test RMSE (dB) | Test MAE (dB) | Test R² |
|---|---:|---:|---:|
| Training-mean baseline | 6.868 | 5.600 | −0.008 |
| Linear regression | 5.087 | 3.946 | 0.447 |
| 32-unit MLP | 2.585 ± 0.110 | 1.959 ± 0.101 | 0.857 ± 0.012 |
| 32-unit MLP with L2 | **2.374 ± 0.111** | **1.789 ± 0.094** | **0.879 ± 0.011** |

MLP values are means ± sample standard deviations across seeds 7, 17 and 27 on one fixed split. They describe training variability, not confidence intervals for new datasets.

![Predicted versus measured noise and residual errors](outputs/final/test_diagnostics.png)

The figure shows the predefined seed-7 run of the selected model family. The table averages three runs. Several of the highest measured noise levels are underpredicted.

## What was compared

- A constant training-mean predictor and ordinary linear regression.
- A PyTorch network with two 32-unit hidden layers, with and without L2 regularisation.
- A controlled size experiment reducing both hidden layers to 16 units.
- Validation-selected checkpoints and a final held-out evaluation.

The 16-unit regularised network had mean validation RMSE of **3.004 dB**, compared with **2.698 dB** for the 32-unit version. The larger regularised configuration was selected before viewing test scores.

## Data and evaluation

The [UCI Airfoil Self-Noise dataset](https://doi.org/10.24432/C5VW2C) has 1,503 rows, five physical inputs and one continuous target: scaled sound pressure level in dB.

Inputs: frequency, angle of attack, chord length, free-stream velocity and suction-side displacement thickness.

Rows sharing the four non-frequency inputs stay together in inferred condition groups. Training, validation and test sets contain **891, 291 and 321 rows**, respectively. Scalers are fitted on training rows only. These inferred groups reduce leakage from related frequency sweeps but do not guarantee complete experimental independence.

Training uses MSE, the Adam optimiser (learning rate 0.001), batches of 64 and 400 epochs. The lowest-validation-error checkpoint is retained; training continues to the full budget for diagnostic curves. The L2 coefficient is 0.001.

## Read the findings

- [Full report](REPORT.md): method, model-size experiment, results, overfitting discussion and limitations.
- [Final per-run metrics](outputs/final/metrics.csv) and [summary](outputs/final/summary.json).
- [Original development results](outputs/my_development/summary.json).
- [Smaller-network results](outputs/smaller_network/summary.json).
- [Data attribution](data/README.md).

## Run the project

Use a Python environment with compatible PyTorch wheels. The saved runs used Python 3.13.15 and PyTorch 2.11.0+cpu; versions are recorded in each `run.json`. The dependency file pins the other main libraries and allows compatible PyTorch 2.x releases. Exact numerical results may vary across environments. CPU is sufficient.

From the project folder, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Reproduce development runs in fresh folders:

```bash
python train.py --out outputs/reproduction32
python train_smaller.py --out outputs/reproduction16
```

The small-network script was reconstructed during packaging by changing only the architecture and its metadata from the final script. The original experiment used the same manual changes in Colab. The final training script and recorded outputs are preserved.

For reproduction of the completed final evaluation:

```bash
python train.py --evaluate-test --out outputs/reproduction_final
```

The supplied test results are already public here. Reproducing them does not create a fresh untouched test set. Further test-informed development would require new independent evaluation evidence.

No Colab notebook is required: the scripts run the complete experiments. Each run saves metrics, histories, plots, neural-network weights, split indices and preprocessing statistics. The final run also saves test predictions.

## Limitations

One small dataset, one grouped split and a limited airfoil family cannot establish operational aircraft reliability. Three initialisations do not measure uncertainty over alternative datasets. The comparison covers a small set of models and training settings, not every competitive regression method. Residuals show some sizeable errors at high measured noise levels.

## Contribution and assistance

This project was completed as a guided learning project. AI tools were used for development support, documentation and code review. I personally ran and evaluated the experiments, modified the model architectures and regularisation settings, compared validation performance, selected the final model, analysed the results and verified the reported outputs.

Dataset attribution and its CC BY 4.0 terms are in [data/README.md](data/README.md). This project is not affiliated with NASA or Airbus.
