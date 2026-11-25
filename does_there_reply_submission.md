# Does There Reply? - Phase 2b Submission Response

## Task

Given a short conversation or reply context, predict whether the intended recipient replies (`1`) or does not reply (`0`). This is a binary classification task represented internally by a 0/1 regression-style score.

## Approach

I used three approaches:

- **Ridge-TFIDF**: word and character TF-IDF features followed by Ridge regression, thresholded at 0.5.
- **BERT-MLP**: frozen `bert-base-uncased` embeddings followed by a compact MLP trained with binary cross-entropy logits.
- **Mistral zero-shot**: local Mistral prompting for a direct 0/1 decision without supervised training.

### Preprocessing and evaluation

For the Ridge and neural baselines, missing text was replaced with an empty string. Ridge used sublinear TF-IDF plus word and character n-grams. The neural approach embedded text using the frozen BERT `[CLS]`/first-token representation and then trained a one-output MLP.

The fixed supplied validation set has 4 rows, while training has 12 rows. I reported both validation MAE and macro-F1. For submission, Ridge and the MLP converted continuous scores to 0/1 using a threshold of 0.5. Ridge was retrained on the combined train and validation rows before test prediction.

Mistral received a binary instruction and parsed the response as exactly `0` or `1`. It was tested through the local Ollama HTTP API over every validation and test row.

### Hyperparameter trials and effects

| Approach | Setting | Value used | Effect on outcome |
|---|---|---:|---|
| Ridge | `RIDGE_ALPHA` | 10.0 | Controls coefficient shrinkage; higher values reduce variance and can suppress sparse TF-IDF noise. It was not swept in a recorded run. |
| Ridge | `WORD_NGRAM_RANGE` | (1,2) | Allows words and adjacent-word phrases to contribute. |
| Ridge | `CHAR_NGRAM_RANGE` | (3,5) | Adds character sequences for spelling and stylistic patterns. |
| Ridge | `MIN_DF` | 1 | Keeps all observed features, including rare words. |
| Ridge/MLP | `THRESHOLD` | 0.5 | Converts the continuous prediction into 0/1. Lower values increase positive recall; higher values increase positive precision. |
| BERT-MLP | `BATCH_SIZE` | 4 | Small batches give more frequent updates, useful for the 12-row training set, but introduce higher gradient noise. |
| BERT-MLP | `LEARNING_RATE` | 2e-4 | With frozen BERT features, this controlled only the small MLP update. |
| BERT-MLP | `WEIGHT_DECAY` | 1e-4 | Mild regularization for MLP parameters. |
| BERT-MLP | `HIDDEN_DIMS` | (128,32) | 128 units then 32 units provide capacity without a large classifier. |
| BERT-MLP | `DROPOUT` | 0.20 | Reduces overfitting; the second layer uses 0.10. |
| BERT-MLP | `MAX_EPOCHS` | 100 | Maximum training duration; early stopping ended training at epoch 16. |
| BERT-MLP | `PATIENCE` | 15 | Stops after 15 epochs without a better validation MAE. |
| Mistral | `TEMPERATURE` | 0.0 | Produces a stable, minimally random answer. |
| Mistral | `NUM_PREDICT` | 3 | Allows the final integer without permitting a long explanation. |

The small validation set makes both macro-F1 and MAE unstable; multiple random splits or repeated evaluation would be needed to estimate the hyperparameters reliably.

### BERT-MLP hyperparameter tuning

The BERT-MLP was selected for submission because it achieved the recorded validation score of MAE 0.2500 and macro-F1 0.4286, while providing a trained neural approach. The selected configuration is:

- Learning rate: `2e-4`
- Weight decay: `1e-4`
- Hidden dimensions: `(128, 32)`
- Dropout: `0.20`
- Batch size: `4`
- Patience: `15`
- Maximum epochs: `100`
- Classification threshold: `0.5`

Three additional configurations were defined as post-selection trials:

| Trial | Learning rate | Hidden dimensions | Dropout | Batch size | Weight decay | Development result |
|---|---:|---|---:|---:|---:|---|
| DTR-T1 | 1e-4 | (128,32) | 0.20 | 8 | 1e-4 | MAE 0.250000; macro-F1 0.428571 |
| DTR-T2 | 5e-4 | (64,16) | 0.30 | 4 | 1e-3 | MAE 0.250000; macro-F1 0.428571 |
| DTR-T3 | 2e-4 | (256,64) | 0.40 | 2 | 1e-4 | MAE 0.250000; macro-F1 0.428571 |

These trials keep frozen BERT embeddings, binary cross-entropy logits, threshold=0.5, seed=42, and validation-MAE checkpoint selection unchanged. Because only four validation rows are available, small metric changes should not be considered reliable without repeated splits or a larger development set.

## Runs

Replace the Kaggle public-score cells below with the values shown on the Kaggle leaderboard.

| run ID | Approach | Dev score | Public leaderboard score |
|---|---|---|---|
| TBD-run1 | Ridge-TFIDF, alpha=10, threshold=0.5 | Validation MAE/F1: not recorded in the saved notebook output | TBD |
| TBD-run2 | BERT-MLP, lr=2e-4, weight_decay=1e-4, hidden=(128,32), dropout=0.20, threshold=0.5 | MAE 0.2500; macro-F1 0.4286 | TBD |
| TBD-run3 | Mistral mistral, temperature=0.0, num_predict=3 | MAE 0.2500; macro-F1 0.4286 | TBD |
| TBD-run4 | BERT-MLP DTR-T1, lr=1e-4, hidden=(128,32), dropout=0.20, batch=8, weight_decay=1e-4 | MAE 0.250000; macro-F1 0.428571 | TBD |
| TBD-run5 | BERT-MLP DTR-T2, lr=5e-4, hidden=(64,16), dropout=0.30, batch=4, weight_decay=1e-3 | MAE 0.250000; macro-F1 0.428571 | TBD |
| TBD-run6 | BERT-MLP DTR-T3, lr=2e-4, hidden=(256,64), dropout=0.40, batch=2, weight_decay=1e-4 | MAE 0.250000; macro-F1 0.428571 | TBD |

The file `submission_ridge.csv`, `submission_nn.csv`, and `submission_mistral_zero_shot.csv` correspond to the three approaches.

## Reflection

BERT-MLP and Mistral zero-shot tied on the available validation result: MAE 0.2500 and macro-F1 0.4286. The neural model stopped after 16 epochs, so it was still improving or tied but required early stopping. Mistral performed competitively without label training, but the four-row validation set is too small to conclude that it generally generalizes better.

The hyperparameter trials test three distinct forms of capacity and regularization. DTR-T1 reduces the learning rate, DTR-T2 increases regularization while reducing network size, and DTR-T3 increases network capacity while also strengthening dropout. The selected base configuration remains preferable unless one of these trials improves both development MAE and macro-F1.

Ridge is the simpler conservative baseline, but its saved output did not include a validation score in the executed notebook file, so it cannot be compared numerically here. This task benefits from preserving both style and conversational turn context rather than relying only on surface wording.

## Code

- Repository: https://github.com/bradm1814/ai-masters-workspace
- Ridge notebook: [DoesThereply_ridge.ipynb](./doesthereply/DoesThereply_ridge.ipynb)
- Neural-network notebook: [DoesThereply_NN.ipynb](./doesthereply/DoesThereply_NN.ipynb)
- Mistral notebook: [DoesThereply_mistral_zero_shot.ipynb](./doesthereply/DoesThereply_mistral_zero_shot.ipynb)

The settings listed in the hyperparameter table reproduce each approach. The public run IDs must be copied from the Kaggle submission descriptions into the run table.

