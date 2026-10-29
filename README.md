# SentimentScope — a Transformer Built From Scratch for Sentiment Analysis

A GPT-style transformer classifier, implemented from first principles in PyTorch and trained
from random initialisation, that decides whether an IMDB film review is positive or negative.

**No pre-trained model weights are used anywhere in this project.** The attention mechanism,
the multi-head wrapper, the feed-forward block, the transformer block and the classification
head are all written out in the notebook. The only thing borrowed from a pre-trained model is
the `bert-base-uncased` *tokenizer* — the vocabulary and the text-to-token-id mapping, not any
learned weights.

Completed as a project of the **AWS AI Programmer course**. It is not affiliated with,
endorsed by, or published by AWS, Udacity or Hugging Face.

---

## Result

| | |
|---|---|
| **Test accuracy** | **76.25%** |
| Project target | > 75% |
| Validation accuracy, untrained | 49.00% |
| Validation accuracy after 3 epochs | 78.00% |

The untrained figure is worth keeping in view: at random initialisation the model sits at
chance on a balanced two-class problem, which is what 49% means. Everything above that was
learned by the architecture in the notebook, not inherited from a pre-trained checkpoint.

| Epoch | Validation accuracy |
|---|---|
| — (initialisation) | 49.00% |
| 1 | 70.80% |
| 2 | 75.20% |
| 3 | 78.00% |

## The data

The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al.,
ACL 2011): 50,000 IMDB reviews, exactly balanced.

| Split | Reviews |
|---|---|
| Train | 22,500 |
| Validation | 2,500 |
| Test | 25,000 |

The 25,000 supplied training reviews are shuffled with a fixed seed and split 90/10 into
train and validation; the supplied 25,000 test reviews are left untouched.

Exploratory analysis in the notebook confirms the labels are balanced 12,500 / 12,500 and
shows the review-length distribution: a mean of 234 words, a median of 174, and a long tail
out to 2,470. That tail matters — the model truncates to a 128-token context, so a majority
of reviews are cut short, and the accuracy has to be read with that in mind.

### Getting the data

The dataset is ~84 MB compressed and is not committed to this repository. Download it and
unpack it next to the notebook:

```bash
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

That produces the structure the notebook expects:

```
aclImdb/
├── train/
│   ├── pos/     12,500 .txt files
│   └── neg/     12,500 .txt files
└── test/
    ├── pos/     12,500 .txt files
    └── neg/     12,500 .txt files
```

## The model

Built up one component at a time in the notebook, each verified against a dummy input before
the next is added:

| Component | What it does |
|---|---|
| `AttentionHead` | Scaled dot-product attention — query, key and value projections, scaled scores, softmax weights, weighted sum of values |
| `MultiHeadAttention` | Several heads in parallel, concatenated and projected back to the embedding width |
| `FeedForward` | Position-wise expansion and projection with a non-linearity between |
| `Block` | Multi-head attention and feed-forward, each with a residual connection and layer normalisation |
| `DemoGPT` | Token and positional embeddings, a stack of blocks, mean pooling across the sequence, and a two-class output head |

### Configuration

| Parameter | Value |
|---|---|
| Embedding width (`d_embed`) | 128 |
| Transformer blocks (`layers_num`) | 4 |
| Attention heads per block (`heads_num`) | 4 |
| Head size | 32 (4 × 32 = 128, matching `d_embed`) |
| Context length | 128 tokens |
| Dropout | 0.1 |
| Vocabulary | `bert-base-uncased`, ~30,522 tokens |
| Output classes | 2 |

### Training

| Setting | Value |
|---|---|
| Epochs | 3 |
| Optimiser | AdamW, learning rate 3e-4 |
| Loss | Cross-entropy |
| Batch size | 32 |
| Steps per epoch | 704 |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook SentimentScope.ipynb
```

Then download the dataset as above and run the cells in order.

**A GPU is strongly recommended.** The run that produced the results above was executed on
CUDA; the notebook falls back to CPU automatically, but three epochs over 22,500 reviews will
take considerably longer there.

## Repository contents

```
.
├── SentimentScope.ipynb    the complete project: data loading, EDA, tokenization,
│                           DataLoader, model implementation, training, evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook is committed **with its outputs intact**, so the results, the training log and
the two exploratory plots are readable on GitHub without running anything.

## Status and reproducibility

- All figures in this README are read directly from the committed notebook's own outputs.
  Nothing is estimated or reconstructed.
- The notebook has **not been re-executed** while preparing this repository — the dataset is
  an 84 MB external download and the original run used a GPU. The outputs you see are from
  the author's original run.
- Re-running will not reproduce these numbers exactly. The train/validation split is seeded,
  but model initialisation, dropout and data-loader shuffling are not, so expect accuracy in
  the same region rather than to the decimal place.

## License

No license file is included, and that is a deliberate decision rather than an oversight. The
notebook is built on a course-supplied template: the narrative markdown, the section
structure and the class skeletons come from the course, and my implementations fill them in.
Applying an open-source license to the whole would assert redistribution rights over the
template that I do not hold.

The IMDB dataset carries its own terms from the authors, and is not redistributed here in any
case.

If you want this repository to carry a license, the question to settle first is whether the
course materials permit redistribution of the template, and under what terms.

## Acknowledgements

- Project template and learning objectives: the **AWS AI Programmer course**.
- Dataset: Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
  *Learning Word Vectors for Sentiment Analysis.* Proceedings of the 49th Annual Meeting of
  the Association for Computational Linguistics, 142–150.
- Tokenizer: `bert-base-uncased` via Hugging Face `transformers`.
