# 💀 The Elsinore Oracle — LSTM vs GRU Next-Word Prediction on *Hamlet*

A deep learning project comparing **LSTM** and **GRU** recurrent architectures on a next-word prediction task, trained on the full text of Shakespeare's *Hamlet*. The two trained models are wrapped in a Streamlit app so their predictions can be inspected side by side, but the core of this project is the modeling work itself.

---

## Table of contents

- [Project goal](#project-goal)
- [Dataset](#dataset)
- [Preprocessing pipeline](#preprocessing-pipeline)
- [Model architectures](#model-architectures)
  - [Why compare LSTM and GRU](#why-compare-lstm-and-gru)
- [Training](#training)
- [Inference / prediction logic](#inference--prediction-logic)
- [LSTM vs GRU — results](#lstm-vs-gru--results)
- [Project structure](#project-structure)
- [Setup & running](#setup--running)
- [The demo app](#the-demo-app)
- [Troubleshooting](#troubleshooting)
- [Possible extensions](#possible-extensions)

---

## Project goal

Given an incomplete sentence from *Hamlet*, predict the most likely next word. The project trains **two separate recurrent neural networks** — one using **LSTM** cells, one using **GRU** cells — on the same corpus, same vocabulary, and same training setup, so that the *only* meaningful variable between them is the recurrent cell type. This makes it possible to compare:

- how each architecture handles long-range dependencies in Early Modern English text
- training behavior (convergence speed, loss curves)
- parameter count and model size
- qualitative differences in predictions on the same input

## Dataset

- **Source text:** the full play text of *Hamlet, Prince of Denmark* (commonly sourced from the NLTK Gutenberg corpus, `shakespeare-hamlet.txt`, or an equivalent plain-text edition).
- **Size:** ~30,000+ words of Early Modern English, including stage directions and speaker labels.
- **Why Hamlet:** a single, internally consistent corpus with a distinctive, archaic vocabulary and syntax — small enough to train quickly, large enough to produce a non-trivial vocabulary and meaningful sequences.

## Preprocessing pipeline

1. **Cleaning** — lowercasing, stripping punctuation/formatting artifacts specific to the source text.
2. **Tokenization** — a Keras `Tokenizer` is fit on the full corpus, mapping every unique word to an integer index. This fitted tokenizer is saved as `tokenizer.pickle` so inference can reuse the exact same word↔index mapping used during training.
3. **Sequence generation** — the text is converted into overlapping **n-gram input sequences** per line (e.g. for the line "to be or not to be", generate `[to, be]`, `[to, be, or]`, `[to, be, or, not]`, ... etc.), so the model learns to predict the next token from every possible prefix.
4. **Padding** — all sequences are padded (pre-padding) to a fixed `max_sequence_len`, so they can be batched.
5. **Split into X / y** — the last token of each padded sequence becomes the label `y` (one-hot encoded across the vocabulary), and everything before it becomes the input `X`.

## Model architectures

Both models share the same overall shape — an `Embedding` layer feeding into one or more recurrent layers, ending in a `Dense` softmax layer over the vocabulary — differing only in the recurrent cell:

**LSTM model** (`next_word_lestm.h5`)
```
Embedding(vocab_size, embedding_dim, input_length=max_sequence_len-1)
LSTM(units, return_sequences=...)
[optional Dropout / additional LSTM layer]
Dense(vocab_size, activation="softmax")
```

**GRU model** (`next_word_GRU_RNN_1.h5`)
```
Embedding(vocab_size, embedding_dim, input_length=max_sequence_len-1)
GRU(units, return_sequences=...)
[optional Dropout / additional GRU layer]
Dense(vocab_size, activation="softmax")
```

### Why compare LSTM and GRU

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (input, forget, output) | 2 (update, reset) |
| Internal state | separate cell state + hidden state | single hidden state |
| Parameters (relative) | more, for the same unit count | fewer — typically ~25% less |
| Training speed (relative) | slower per epoch | usually faster per epoch |
| Known strengths | often better at capturing longer-range dependencies | often converges faster, competitive accuracy on smaller datasets |

On a corpus the size of a single play, the GRU's smaller parameter count can be an advantage against overfitting, while the LSTM's extra gating may help it track dependencies across longer, more convoluted Shakespearean sentence structure. This project trains both under identical conditions to see which trade-off actually wins out on this data.

## Training

- **Loss:** categorical cross-entropy (multi-class classification over the vocabulary)
- **Optimizer:** Adam (typical default; adjust to match your actual run)
- **Metric:** accuracy (next-word match)
- **Callbacks:** commonly `EarlyStopping` on validation loss to avoid overfitting on a relatively small corpus

## Inference / prediction logic

Both models are queried with the same function, so predictions are directly comparable:

```python
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
```

Key details:
- `max_sequence_len` is read from each model's own `input_shape`, so the LSTM and GRU can have been trained with different sequence lengths without breaking inference.
- Prediction is **greedy** — it always takes `argmax` over the output distribution, i.e. the single most probable next word, not a sampled one.
- For multi-word generation, the predicted word is appended back onto the input and the function is called again, run independently for each model — a simple autoregressive loop.

## LSTM vs GRU — results

This is where the interesting part of the comparison goes. Suggested things to record and discuss here once you've run both models side by side on a shared set of test lines:

- On which inputs do the two models **agree** vs **disagree** on the next word?
- Does one model tend to produce more "generic" high-frequency words while the other produces more specific/contextual ones?
- Does either model degrade faster over multi-word (autoregressive) generation — e.g. falling into repetition loops sooner?
- How do validation loss curves compare — does one overfit sooner given the corpus size?

> _Add your actual qualitative and quantitative findings here — this section is the heart of the write-up._

## Project structure

```
.
├── app.py                      # Streamlit demo comparing both models live
├── next_word_lestm.h5          # trained LSTM model (Keras)
├── next_word_GRU_RNN_1.h5      # trained GRU model (Keras)
├── tokenizer.pickle            # Tokenizer fitted on the Hamlet corpus (shared by both models)
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup & running

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt`:
```
streamlit
tensorflow
numpy
```

Make sure `next_word_lestm.h5`, `next_word_GRU_RNN_1.h5`, and `tokenizer.pickle` sit next to `app.py` before launching.

## The demo app

`app.py` is a Streamlit interface (nicknamed *The Elsinore Oracle*) built purely to make the LSTM/GRU comparison interactive:

- Enter or **randomly summon** an incomplete Hamlet line
- Get both models' next-word predictions simultaneously, revealed word by word
- Generate up to 10 words ahead per model and compare where their generated text diverges
- See a quick verdict on whether the two models agree on the very next word

It's a visualization layer over the modeling work above — the interesting engineering is in the two trained models, not the UI.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `The Oracle could not be summoned...` | One of the three artifact files is missing/misnamed | Confirm `next_word_lestm.h5`, `next_word_GRU_RNN_1.h5`, `tokenizer.pickle` are present next to `app.py` |
| `StreamlitAPIException: ... cannot be modified after widget is instantiated` | A widget's session_state key was reassigned after that widget was already drawn this run | Use an `on_click` callback to update the value, rather than assigning inside an `if st.button(...):` block |
| Predictions look repetitive over multiple words | Greedy decoding (always picking argmax) will loop on some inputs | Expected behavior for greedy generation, not a bug — consider adding temperature/top-k sampling as an extension |
| Different `max_sequence_len` errors between models | The two models were trained with different input lengths | This is handled automatically — each model's own `input_shape` is used, but confirm the tokenizer was shared/consistent across both training runs |

## Possible extensions

- Add temperature or top-k/nucleus sampling instead of pure greedy decoding, to compare creative diversity between LSTM and GRU outputs.
- Log perplexity for both models on a shared held-out set of Hamlet lines for a more rigorous quantitative comparison.
- Add a stacked/bidirectional variant of each architecture and extend the comparison table.
- Try the same pipeline on a second Shakespeare play to see if the LSTM/GRU gap changes with more or different training data.

---

*"The rest is silence."* — Hamlet, Act V, Scene II
