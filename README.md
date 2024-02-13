# nanogpt with labels

Nanogpt with labels implemented. Performance on the task of subtracting two single-digit numbers is increased for the comparable hyperparameters.

## subtraction task results

# no_labels
39M parameter model, 512 context window
   0shot_cot: 373
0shot_direct: 294
   nshot_cot: 453
nshot_direct: 376

# with_labels
39M parameter model, 512 (effective) context window
   0shot_cot: 173
0shot_direct: 142
   nshot_cot: 178
nshot_direct: 193

# label_embeddings
39M parameter model, 512 context window, 
   0shot_cot: 421
0shot_direct: 263
   nshot_cot: 382
nshot_direct: 956

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