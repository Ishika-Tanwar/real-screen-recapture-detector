"""
evaluate.py — Run predict.py's model against the full dataset and report accuracy.

This is NOT the same as train.py's validation accuracy (which only checks
a held-out 20% split). This checks every image you have, using the final
ONNX model, to sanity-check before submission.

Usage:
    python evaluate.py
"""

import os
from predict import predict

DATA_DIR = "test_data"
THRESHOLD = 0.5

def evaluate_folder(folder_path, true_label, label_name):
    """true_label: 0 for real, 1 for screen"""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    correct = 0
    wrong_files = []

    for fname in files:
        path = os.path.join(folder_path, fname)
        score = predict(path)
        predicted_label = 1 if score >= THRESHOLD else 0

        if predicted_label == true_label:
            correct += 1
        else:
            wrong_files.append((fname, score))

    acc = correct / len(files) if files else 0
    print(f"{label_name}: {correct}/{len(files)} correct ({acc*100:.1f}%)")

    if wrong_files:
        print(f"  Misclassified:")
        for fname, score in wrong_files:
            print(f"    {fname}  (score={score:.4f})")

    return correct, len(files)


if __name__ == "__main__":
    print("Evaluating model on full dataset...\n")

    real_correct, real_total = evaluate_folder(
        os.path.join(DATA_DIR, "real"), true_label=0, label_name="REAL photos"
    )
    print()
    screen_correct, screen_total = evaluate_folder(
        os.path.join(DATA_DIR, "screen"), true_label=1, label_name="SCREEN photos"
    )

    total_correct = real_correct + screen_correct
    total = real_total + screen_total
    overall_acc = total_correct / total if total else 0

    print(f"\nOverall accuracy: {total_correct}/{total} ({overall_acc*100:.2f}%)")