# Fast Conformer Paper Review

Oct 22, 2026 · @Tohid

## The problem it solves

Fast Conformer (NVIDIA, 2023) makes the dominant speech-recognition encoder about 2.8x faster at the same accuracy, and lets it handle audio up to 11 hours long.

Conformer combines convolution for local acoustic features with self-attention for global context. Its weakness is cost:

- Self-attention scales quadratically with sequence length, which caps audio at about 15 minutes on one A100 GPU.
- The front-end subsampling block is expensive, taking over 20% of forward-pass time for a Large (120M) model.
- Scaling to very large models needs kernel-size changes to stay stable.

## The core ideas

Three changes carry the paper: aggressive early downsampling, linear-cost attention for long audio, and scaling to 1B parameters.

### 1. A redesigned downsampling schema (the main contribution)

The original Conformer reduces the frame rate from 10 ms to 40 ms (4x). Fast Conformer goes to 80 ms (8x) at the start of the encoder, so every attention layer sees sequences half as long and attention cost drops about 4x.

- 8x downsampling up front, instead of the progressive schemes in Efficient Conformer and Squeezeformer, which leave early layers on much longer sequences.
- Depthwise-separable convolutions replace the standard 2D convolutions in the subsampling block.
- Subsampling filters are cut from 512 to 256.
- Convolution kernel size is cut from 31 to 9.

A subtle point: earlier work stopped at 4x because CTC needs the encoder output to be at least as long as the target sequence, and 8x breaks this with character tokens. The authors switch to subword (BPE/SentencePiece) tokens with 128 to 1024 vocabularies. RNNT has no such constraint.

### 2. Linearly scalable attention for long audio

After training, full self-attention is swapped for Longformer-style attention: a local window of 128 frames each side (about 10 seconds) plus one global token. The global token attends to everything, everything attends to it, and it uses separate Q/K/V projections. This makes attention cost linear in length.

Only a short fine-tune is needed (10k steps at a 1e-6 peak learning rate) from a full-attention checkpoint, so nothing is retrained from scratch.

### 3. Scaling to 1B parameters

The core architecture stays unchanged. L (120M), XL (600M) and XXL (1.1B) variants come from widening and deepening only. For XXL, the encoder is pretrained with self-supervised learning (wav2vec 2.0 style) to stabilize training.

## Key results

The headline is efficiency at equal accuracy: roughly 3x less compute, with long-form transcription as the biggest gain.

| Claim | Evidence |
| --- | --- |
| \~2.8x faster encoder, same accuracy | Table 2: 169 to 467 samples/s; WER 5.19 to 4.99 on LibriSpeech test-other |
| \~3x fewer GMACs | Table 4: 143.2 to 48.7 (RNNT) |
| Longer audio on one GPU | Table 3: 15 min to 675 min with limited context (the abstract's "11 hours") |
| Better long-form WER | Table 8: Earnings-21 18.26 to 11.85 with local attention plus global token |
| Speech translation (En-De) | Table 6: 1.66x to 1.84x speedup; RNNT decoder BLEU 23.2 to 27.9 |
| Spoken language understanding | Table 7: beats HuBERT-based systems; only 1.1x speedup |
| Scales to 1B | Table 10: FC-XXL reaches 2.52 on LibriSpeech test-other |

## Strengths

The paper is practical, well ablated and candid about where the speedup does not apply.

- **Practical and simple.** The changes are easy to implement, and the models are open-sourced in NVIDIA NeMo.
- **Careful ablation.** Table 2 adds each change incrementally, so each one's contribution is visible.
- **Honest about limits.** The SLU speedup is only 1.1x, and the authors explain why: the autoregressive Transformer decoder dominates cost, not the encoder.
- **Broad evaluation** across ASR, speech translation, SLU and long-form audio.

## Weaknesses and caveats

The efficiency claims are well supported, but several accuracy and scaling claims are weaker than the headline suggests.

- **Accuracy is equal, not clearly better.** On the 25k-hour set (Table 5), RNNT Fast Conformer is slightly worse on LibriSpeech test-other (3.79 vs 3.74) and Common Voice (8.18 vs 7.87).
- **Long-form comparison is not like for like.** Baselines use 20-second buffered transcription, while the limited-context model processes the whole file at once and gets extra fine-tuning. Part of the gain is the long-form training, not the architecture alone.
- **XXL results are not reproducible.** They rely on 40k hours of internal, unreleased data (ASR Set++). SSL pretraining also confounds the scaling claim, so it is unclear how much comes from architecture versus pretraining.
- **Speed numbers are narrow.** They come from one GPU type (A100), specific batch sizes and 20-second clips. Real streaming or on-device latency is not evaluated.
- **No streaming or causal analysis,** even though the introduction stresses industry streaming use.
- **Minor presentation issues.** Table 8's caption mentions four versions but lists three, and the Vaswani reference appears three times.

## Why it matters

Aggressive early downsampling is the cheapest efficiency win in speech encoders, because speech frames are highly redundant at 10 ms.

Fast Conformer became the backbone of many NVIDIA NeMo models, including the Canary and Parakeet ASR families. Its Longformer-style trick also shows that long-context ability can be retrofitted onto a pretrained model with minimal fine-tuning.

**Source:** [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (arXiv:2305.05084)](https://arxiv.org/abs/2305.05084), v6, 30 Sep 2023.
