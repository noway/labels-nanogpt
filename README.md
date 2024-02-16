# nanogpt with labels

Nanogpt with labels implemented. Performance on the task of subtracting two single-digit numbers is increased for the comparable hyperparameters.

## subtraction task results

ranking sum, lower is better

### `no_labels`
39M parameter model, 512 context window
| eval_type    | sum |
| ------------ |-----|
|    0shot_cot | 373 |
| 0shot_direct | 294 |
|    nshot_cot | 453 |
| nshot_direct | 376 |

### `with_labels`
39M parameter model, 512 (effective) context window
| eval_type    | sum |
| ------------ |-----|
|    0shot_cot | 173 |
| 0shot_direct | 142 |
|    nshot_cot | 178 |
| nshot_direct | 193 |

### `label_embeddings`
39M parameter model, 512 context window, 
| eval_type    | sum |
| ------------ |-----|
|    0shot_cot | 421 |
| 0shot_direct | 263 |
|    nshot_cot | 382 |
| nshot_direct | 956 |

## training

- 6.5 hours compute budget on 1 H100 for each model
- 132 batch_size for 512 context window, 38 batch_size for 1048 context window

## tokenizer & labeler

- BPE-like tokenizer, vocab_size 1024
- labels are word commonality based + character "type" ([digit, typography, emoji, special, etc](toker.py#L249))

## original hypothesis & findings

The original hypothesis was that the label embedding model must increase eval performance over no labels. This has been proven false - label embedding model has slightly better/worse eval performance, overall mixed. On the other hand, having a label on every second token (`with_labels` model) does increase eval performance for the comparable hyperparameters. The intuition here is that the extra token is a chance for the model to "think".

## caveats

- The model may be memorizing the substraction result from the dataset, thus becoming more of a recall test rather than calculation test. Although there are instances where the model gets the right result that's not in the dataset.
- No optimal hyperparameter search was done.
- `main` branch is for evals, training is in `real_run`

## weights

[See here](WEIGHTS.md)