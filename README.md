# nanogpt with label embeddings

Nanogpt with label embeddings implemented. Performance on the task of subtracting two single-digit numbers is increased for the comparable hyperparameters.

## subtraction task results

### no labels
39M parameter model, 1024 context window
| eval_type     | correct_count  | all_count  |
| ------------- |:--------------:| ----------:|
| nshot_cot     | 4              | 55         |
| nshot_direct  | 7              | 55         |
| 0shot_cot     | 7              | 55         |
| 0shot_direct  | 9              | 55         |


### labels (every second token)
39M parameter model, 1024 (effective) context window
| eval_type     | correct_count  | all_count  |
| ------------- |:--------------:| ----------:|
| nshot_cot     | 9              | 55         |
| nshot_direct  | 10             | 55         |
| 0shot_cot     | 8              | 55         |
| 0shot_direct  | 6              | 55         |

### label embeddings
39M parameter model, 1024 context window, 
| eval_type     | correct_count  | all_count  |
| ------------- |:--------------:| ----------:|
| nshot_cot     | 13             | 55         |
| nshot_direct  | 10             | 55         |
| 0shot_cot     | 8              | 55         |
| 0shot_direct  | 15             | 55         |

## training

- 40000 iterations on 1 H100
- 18 batch_size for 1024 context window, 4 batch_size for 2048 context window

## tokenizer & labeler

- BPE-like tokenizer, vocab_size 1024
- labels are word commonality based + character "type" ([digit, typography, emoji, special, etc](toker.py#L249))

## label embedding
- [label embeddings are only for input](train.py#L176), similar to BERT

## caveats

- the model may be memorizing the substraction result from the dataset, thus becoming more of a recall test rather than calculation test.
- since the batch_size for 2048 context window is different due to memory constraints, it's harder to directly compare to the 2 other variants.