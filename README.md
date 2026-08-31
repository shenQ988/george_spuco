# GEORGE SpuCoMNIST Experiment

This project compares standard empirical risk minimization (ERM) with group-balanced ERM on SpuCoMNIST, a version of MNIST with an intentionally correlated spurious feature.

The notebook groups the original digits into five target classes: `{0, 1}`, `{2, 3}`, `{4, 5}`, `{6, 7}`, and `{8, 9}`. During training, the spurious feature matches the target class for most examples. This makes it possible for a standard model to learn a shortcut instead of relying only on digit shape.

## Contents

- `assignment_spuco_mnist_erm.ipynb` — dataset setup, ERM baseline, K-means group inference, group-balanced training, and evaluation.
- `erm_training_logs.txt` — verbose log from training the ERM baseline.
- `group_balanced_training_logs.txt` — verbose log from training the group-balanced model.

## Run the experiment

Install the dependencies:

```bash
pip install spuco torch torchvision matplotlib scikit-learn jupyter
```

Then open and run the notebook from top to bottom:

```bash
jupyter notebook assignment_spuco_mnist_erm.ipynb
```

## Main idea

ERM minimizes average loss, so the large majority groups can dominate training. The notebook uses the ERM model's logits to infer two clusters within each target class, then trains `GroupBalanceBatchERM` with balanced sampling across those inferred groups. Performance is reported using both average accuracy and worst-group accuracy, since average accuracy can hide poor performance on rare spurious groups.

The verbose training output is stored in separate files to keep the notebook concise when rendered on GitHub.
