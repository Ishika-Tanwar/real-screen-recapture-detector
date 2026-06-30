## 🔴 Live Demo
👉 [Try it on Hugging Face](https://huggingface.co/spaces/IshikaTanwar/screen-recapture-detector)


# Spot the Fake Photo — Screen Recapture Detector

A lightweight binary classifier that detects whether an image is a **real photo** or a **photo of a screen** (recapture fraud).

There's no object to recognize here — the model has to pick up on subtle cues: screen texture/moiré, glare, and slightly-off colors from the recapture process.

---

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run prediction:**
```bash
python predict.py some_image.jpg
```
Outputs a single float from `0` to `1`:
- `0` = real photo
- `1` = photo of a screen (fraud)

---

## How I did it

I fine-tuned **MobileNetV2** (pretrained on ImageNet) as a binary classifier — small, fast, and a good fit for on-device deployment.

- **Data:** 113 real photos + 107 screen-recapture photos, taken with my own phone across varied lighting, screen types (phone/laptop/monitor), angles, and distances.
- **Training:** Classifier head + last 3 conv blocks of MobileNetV2 fine-tuned, Adam (lr=1e-3), flip/rotation/color-jitter augmentation, 80/20 train/val split, early stopping (patience=6, converged at epoch 4 of 25).
- **Inference:** Exported to ONNX and run via ONNX Runtime, which roughly halved CPU latency vs. raw PyTorch with no change in accuracy.
- **Output:** A single sigmoid score in [0,1] — higher means more likely a screen recapture.

## Accuracy — honest number

**97.7% on my own validation split (44 held-out images from a 220-photo training set).**

To get a more honest read, I also shot a **separate batch of 33 brand-new photos** (16 real + 17 screen) on a different day, that the model never saw during training or validation, and ran them through `predict.py` as a true held-out test:

**96.97% on this independent test set (32/33 correct)** — only one real photo was misclassified (a borderline case scoring 0.78).

I'm still upfront about the limits here: both sets come from me, my phones, and my environments. A genuinely different person/device/setting (which is presumably how this will actually be graded) could behave differently, especially around edge cases like reflective surfaces or unusual angles. But having a true held-out test that the model never touched during training, and seeing it hold up close to validation accuracy, gives me reasonable confidence this isn't just overfitting to my training distribution.

---

## Required numbers

**Latency:** ~40–46 ms/image with ONNX Runtime, measured end-to-end (image load + preprocess + inference) on a laptop CPU (Intel i5, no GPU, Windows 11). No batching, single image at a time. This is roughly 2x faster than the raw PyTorch model (~75-90ms) at identical accuracy — ONNX export removes Python/framework overhead.

**Cost per image:**
- **On-device (phone):** effectively free — MobileNetV2 is small enough to run directly on a user's phone via Core ML / TFLite, no server round-trip, no marginal cost.
- **Cloud server (CPU instance, e.g. AWS t3.medium at ~$0.0416/hr):** at ~47ms/image and one request at a time, that's roughly 76,000 images/hour per instance → **~$0.55 per 1,000 images**, or **~$550 per million images**. This drops further with batching and/or a smaller instance.
- **Assumption:** single-threaded CPU inference, no batching, no autoscaling discount. A production setup would batch requests and could realistically bring this down another 5-10x.

**Trade-off made:** I optimized for small and cheap over maximum accuracy. MobileNetV2 with most of the backbone frozen, plus ONNX export, gives a model that's both small and fast, at some cost to ceiling accuracy versus a larger architecture. Given the assignment's "prefer small, fast, cheap over big and complicated," this felt like the right trade.

---

## What I'd improve with more time

1. **Much larger, more diverse dataset** — 500+ images per class, multiple people's phones/screens/environments, not just mine. This is the single biggest lever on real-world accuracy.
2. **Frequency-domain features** — screen recaptures often show moiré patterns visible in the FFT domain; combining an FFT-based signal with the CNN output (ensemble or extra input channel) could catch cases the CNN misses, especially with the textures/illumination it hasn't seen.
3. **More held-out test data** — collect more photos with different phones/people/environments than training data, to get an even more confident read on generalization.
4. **Quantization (int8)** — the model is already exported to ONNX; quantizing it further could cut latency and size even more for true on-device deployment.

## Keeping it accurate as cheaters adapt

This is fundamentally an arms race: as detection improves, recapture techniques will too (better lighting rigs, anti-glare filters, higher-res displays that produce less moiré). To stay ahead I'd:
- Continuously collect new recapture attempts (especially ones the model gets wrong) and retrain periodically — treat this as an active-learning loop, not a one-time train.
- Avoid relying on any single cue (e.g. moiré alone) since that's the easiest one to defeat with a better screen; combine multiple weak signals (texture, glare, color cast, edge/bezel detection) so no single countermeasure breaks the whole system.
- Monitor the score distribution over time in production — a sudden shift (e.g. average scores creeping down) is a signal that recapture techniques have changed and the model needs retraining.

## Choosing the cutoff score

Rather than a single fixed threshold, I'd pick the cutoff based on the cost of each error type:
- A **false positive** (flagging a real photo as fraud) annoys/blocks a legitimate user.
- A **false negative** (missing a real recapture) lets fraud through.

If the cost of letting fraud through is high, I'd bias the threshold lower (e.g. 0.3 instead of 0.5) to catch more recaptures at the cost of more false flags — and pair this with a graceful UX (e.g. "please retake the photo" rather than an outright ban) so false positives are cheap to recover from. I'd tune this using a precision-recall curve on a representative validation set rather than guessing, and revisit it as the model and the threat model evolve.

---

## File Structure

```
├── data/              ← training photos (real/ and screen/)
├── test_data/          ← held-out test photos, never seen during training
├── train.py            ← training script
├── export_onnx.py      ← converts trained .pth model to ONNX for fast inference
├── evaluate.py          ← runs predict.py against a folder of images and reports accuracy
├── predict.py            ← inference script (main deliverable, uses ONNX Runtime)
├── model.pth              ← trained PyTorch weights
├── model.onnx              ← exported ONNX model (used by predict.py)
└── requirements.txt
```