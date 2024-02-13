# nanogpt with labels

Nanogpt with labels implemented. Performance on the task of subtracting two single-digit numbers is increased for the comparable hyperparameters.

## subtraction task results

### no labels
39M parameter model, 512 context window
| eval_type     | correct_count  | all_count  |
| ------------- |:--------------:| ----------:|
| nshot_cot     | 9              | 55         |
| nshot_direct  | 10             | 55         |
| 0shot_cot     | 10             | 55         |
| 0shot_direct  | 12             | 55         |


### labels (every second token)
39M parameter model, 512 (effective) context window
| eval_type     | correct_count  | all_count  |
| ------------- |:--------------:| ----------:|
| nshot_cot     | 9              | 55         |
| nshot_direct  | 12             | 55         |
| 0shot_cot     | 14             | 55         |
| 0shot_direct  | 13             | 55         |

### label embeddings
39M parameter model, 512 context window, 
| eval_type     | correct_count  | all_count  |
| ------------- |:--------------:| ----------:|
| nshot_cot     | 7              | 55         |
| nshot_direct  | 10             | 55         |
| 0shot_cot     | 10             | 55         |
| 0shot_direct  | 13             | 55         |

## training

- 6.5 hours iterations on 1 H100
- 132 batch_size for 512 context window, 38 batch_size for 1048 context window

## tokenizer & labeler

- BPE-like tokenizer, vocab_size 1024
- labels are word commonality based + character "type" ([digit, typography, emoji, special, etc](toker.py#L249))

## label embedding
- [label embeddings are only for input](train.py#L176), similar to BERT

## caveats

- the model may be memorizing the substraction result from the dataset, thus becoming more of a recall test rather than calculation test.
- since the batch_size for 2048 context window is different due to memory constraints, it's harder to directly compare to the 2 other variants.