import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
STATES_DIR = PROJECT_ROOT / "states"

with (DATA_DIR / "flip_results.json").open("r", encoding="utf-8") as f:
    items = json.load(f)
assert all(item["correct"] != item["wrong"] for item in items)
assert all(
    item["correct"] in ("A", "B", "C", "D")
    and item["wrong"] in ("A", "B", "C", "D")
    for item in items
), "Invalid answer label found"

ANSWER_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
correct_labels = np.array([ANSWER_TO_ID[it["correct"]] for it in items])
suggested_labels = np.array([ANSWER_TO_ID[it["wrong"]] for it in items])
CONDITIONS = ["user_belief", "textbook", "answer_key", "unattributed"]

hidden_states = {"neutral": np.load(STATES_DIR / "neutral.npy")}
for condition in CONDITIONS:
    hidden_states[condition] = np.load(STATES_DIR / f"{condition}_wrong.npy")
expected_shape = hidden_states["neutral"].shape

assert len(expected_shape) == 3, "Expected 3D hidden-state arrays"
assert expected_shape[0] == len(items), "Question count mismatch"

for condition, states in hidden_states.items():
    assert states.shape == expected_shape, (
        f"{condition}: expected {expected_shape}, got {states.shape}"
    )
    

NUM_QUESTIONS = len(items)
rng = np.random.RandomState(0)
question_indices = rng.permutation(NUM_QUESTIONS)

# split by QUESTION, so no question is in both
train_end = NUM_QUESTIONS // 2
validation_end = 3 * NUM_QUESTIONS // 4

train_indices = question_indices[:train_end]
validation_indices = question_indices[train_end:validation_end]
test_indices = question_indices[validation_end:]

neutral_correct = np.array([it["pred_neutral"] == it["correct"] for it in items])

# identify incorrect flips and misleading-suggestion agreement
flipped = {}
followed_suggestion = {}

for condition in CONDITIONS:
    incorrect = np.array([
        item[f"pred_{condition}_wrong"] != item["correct"]
        for item in items
    ])

    followed = np.array([
        item[f"pred_{condition}_wrong"] == item["wrong"]
        for item in items
    ])

    flipped[condition] = neutral_correct & incorrect
    followed_suggestion[condition] = neutral_correct & followed

# test indices for each outcome
flipped_test_indices = {
    condition: test_indices[flipped[condition][test_indices]]
    for condition in CONDITIONS
}

followed_test_indices = {
    condition: test_indices[followed_suggestion[condition][test_indices]]
    for condition in CONDITIONS
}

for condition in CONDITIONS:
    print(
        f"{condition}: "
        f"{len(flipped_test_indices[condition])} incorrect flips, "
        f"{len(followed_test_indices[condition])} followed suggestions"
    )


def train_probe(features, labels):
    """Train a linear probe to predict answer labels from hidden states."""
    probe = make_pipeline(
            StandardScaler(), 
            LogisticRegression(C=0.05, max_iter=500)
    )
    probe.fit(features, labels)
    return probe

def evaluate_probe(probe, features, labels):
    """Evaluate classification accuracy on held-out examples."""
    if len(labels) == 0:
        return None

    predictions = probe.predict(features)
    return float(accuracy_score(labels, predictions))

# Skip index 0 (embedding output); probe transformer layers only.
layers = list(range(1, hidden_states["neutral"].shape[1]))

result_keys = ["neutral_probe_accuracy"] + [
    f"{condition}_{metric}"
    for condition in CONDITIONS
    for metric in (
        "neutral_matched",
        "transfer",
        "in_cond",
        "hint",
        "hint_all",
        "shuffled",
    )
]
results = {k: {} for k in result_keys}

neutral_correct_test_indices = test_indices[neutral_correct[test_indices]]
shuffled_train_labels = rng.permutation(correct_labels[train_indices])

neutral_correct_validation_indices = validation_indices[
    neutral_correct[validation_indices]
]

if len(neutral_correct_validation_indices) == 0:
    raise ValueError("No neutral-correct questions in the validation set.")

validation_accuracy = {}

for layer in layers:
    X_neutral = hidden_states["neutral"][:, layer].astype(np.float32)

    # Train a probe on neutral hidden states
    neutral_probe = train_probe(
        X_neutral[train_indices],
        correct_labels[train_indices]
    )

    # Validation is used to select the layer later
    validation_accuracy[layer] = evaluate_probe(
        neutral_probe,
        X_neutral[neutral_correct_validation_indices],
        correct_labels[neutral_correct_validation_indices]
    )

    # Train a negative control using shuffled labels
    shuffled_probe = train_probe(
        X_neutral[train_indices],
        shuffled_train_labels
    )

    # Neutral accuracy across all neutral-correct test questions
    results["neutral_probe_accuracy"][layer] = evaluate_probe(
        neutral_probe,
        X_neutral[neutral_correct_test_indices],
        correct_labels[neutral_correct_test_indices]
    )

    for condition in CONDITIONS:
        X_condition = hidden_states[condition][:, layer].astype(np.float32)

        # Questions where the model followed the misleading suggestion
        target_indices = followed_test_indices[condition]

        # 1. Neutral probe evaluated on the matched neutral states
        results[f"{condition}_neutral_matched"][layer] = evaluate_probe(
            neutral_probe,
            X_neutral[target_indices],
            correct_labels[target_indices]
        )

        # 2. Same probe evaluated on pressured states
        results[f"{condition}_transfer"][layer] = evaluate_probe(
            neutral_probe,
            X_condition[target_indices],
            correct_labels[target_indices]
        )

        # 3. Shuffled-label negative control
        results[f"{condition}_shuffled"][layer] = evaluate_probe(
            shuffled_probe,
            X_condition[target_indices],
            correct_labels[target_indices]
        )

        # 4. Probe trained on pressured states to predict correct answers
        condition_probe = train_probe(
            X_condition[train_indices],
            correct_labels[train_indices]
        )

        results[f"{condition}_in_cond"][layer] = evaluate_probe(
            condition_probe,
            X_condition[target_indices],
            correct_labels[target_indices]
        )

        # 5. Probe trained to predict the misleading answer
        hint_probe = train_probe(
            X_condition[train_indices],
            suggested_labels[train_indices]
        )

        results[f"{condition}_hint"][layer] = evaluate_probe(
            hint_probe,
            X_condition[target_indices],
            suggested_labels[target_indices]
        )

        # 6. Evaluate hint probe on all test questions, not just followed suggestions
        results[f"{condition}_hint_all"][layer] = evaluate_probe(
            hint_probe,
            X_condition[test_indices],
            suggested_labels[test_indices]
        )

    print("layer", layer, flush=True)

RESULTS_DIR.mkdir(exist_ok=True)
with (RESULTS_DIR / "probe_results.json").open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, allow_nan=False)

# Select a single layer using validation accuracy
selected_layer = max(
    validation_accuracy,
    key=validation_accuracy.get
)

print(f"\nSelected layer: {selected_layer}")
print(f"Validation accuracy: {validation_accuracy[selected_layer]:.1%}")


# Report probe accuracy with 95% confidence intervals.
for condition in CONDITIONS:
    followed_test_count = len(followed_test_indices[condition])

    print(f"\n{condition} (n={followed_test_count})")

    for probe_type in (
        "neutral_matched",
        "transfer",
        "in_cond",
        "hint",
        "hint_all",
        "shuffled",
    ):
        probe_accuracy = results[f"{condition}_{probe_type}"][selected_layer]

        if probe_accuracy is None:
            print(f"  {probe_type}: N/A")
            continue

        # Hint_all evaluates every test question; other probes use followed questions.
        evaluation_count = (
            len(test_indices)
            if probe_type == "hint_all"
            else followed_test_count
        )

        correct_prediction_count = round(probe_accuracy * evaluation_count)

        confidence_interval = binomtest(
            correct_prediction_count,
            evaluation_count
        ).proportion_ci(
            confidence_level=0.95,
            method="wilson"
        )

        print(
            f"  {probe_type}: {probe_accuracy:.1%} "
            f"(95% CI: {confidence_interval.low:.1%}–"
            f"{confidence_interval.high:.1%})"
        )

# Control: train only on questions where the model stayed correct under pressure.
print("\nControl: trained on stayed-correct questions")
for condition in CONDITIONS:
    condition_states = hidden_states[condition][:, selected_layer].astype(np.float32)

    stayed_correct = neutral_correct & np.array([
        item[f"pred_{condition}_wrong"] == item["correct"]
        for item in items
    ])

    stayed_train_indices = train_indices[stayed_correct[train_indices]]
    followed_indices = followed_test_indices[condition]

    if len(np.unique(correct_labels[stayed_train_indices])) < 2:
        print(f"  {condition}: insufficient training classes")
        continue

    stayed_correct_probe = train_probe(
        condition_states[stayed_train_indices],
        correct_labels[stayed_train_indices]
    )

    stayed_correct_accuracy = evaluate_probe(
        stayed_correct_probe,
        condition_states[followed_indices],
        correct_labels[followed_indices]
    )

    accuracy_text = (
        f"{stayed_correct_accuracy:.1%}"
        if stayed_correct_accuracy is not None
        else "N/A"
    )

    print(
        f"  {condition}: {accuracy_text} "
        f"(train n={len(stayed_train_indices)}, "
        f"test n={len(followed_indices)})"
    )

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

for ax, condition in zip(axes.flat, CONDITIONS):
    for metric, label in [
        ("neutral_matched", "matched neutral baseline"),
        ("transfer", "neutral-trained → pressured"),
        ("in_cond", "trained in condition"),
        ("hint", "predicts suggested wrong answer"),
        ("hint_all", "predicts hint (all test questions)"),
        ("shuffled", "shuffled-label control"),
    ]:
        accuracies = [
            results[f"{condition}_{metric}"][layer]
            for layer in layers
        ]
        ax.plot(layers, accuracies, label=label)

    # uniform random-guessing reference
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=0.8, 
               label="Uniform chance (25%)")

    # layer selected using validation data
    ax.axvline(selected_layer, color="gray", linestyle="--", linewidth=0.8,
               label=f"Layer {selected_layer} (neutral-validation)")

    ax.axhline(
        1 / 3,
        color="orange",
        linestyle=":",
        linewidth=0.8,
        label="Hint-exclusion baseline (33.3%)"
    )

    ax.set_title(
        f"{condition} (n={len(followed_test_indices[condition])})"
    )
    ax.set_xlabel("Layer")

axes[0, 0].set_ylabel("Probe accuracy")
axes[1, 0].set_ylabel("Probe accuracy")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(RESULTS_DIR / "probe_by_layer.png", dpi=150)
plt.close(fig)



# Control across all layers, with validation-based layer selection.
control_test = {condition: {} for condition in CONDITIONS}
control_validation = {condition: {} for condition in CONDITIONS}

validation_followed_indices = {
    condition: validation_indices[
        followed_suggestion[condition][validation_indices]
    ]
    for condition in CONDITIONS
}

stayed_correct_train_indices = {}

for condition in CONDITIONS:
    stayed_correct = neutral_correct & np.array([
        item[f"pred_{condition}_wrong"] == item["correct"]
        for item in items
    ])

    stayed_correct_train_indices[condition] = train_indices[
        stayed_correct[train_indices]
    ]

    if len(np.unique(correct_labels[stayed_correct_train_indices[condition]])) < 2:
        raise ValueError(f"{condition}: insufficient training classes")

    if len(validation_followed_indices[condition]) == 0:
        raise ValueError(f"{condition}: no followed-suggestion validation questions")

    if len(followed_test_indices[condition]) == 0:
        raise ValueError(f"{condition}: no followed-suggestion test questions")

print("\nControl across all layers (stayed-correct training)")

for layer in layers:
    for condition in CONDITIONS:
        condition_states = hidden_states[condition][:, layer].astype(np.float32)

        control_probe = train_probe(
            condition_states[stayed_correct_train_indices[condition]],
            correct_labels[stayed_correct_train_indices[condition]]
        )

        test_indices_followed = followed_test_indices[condition]
        validation_indices_followed = validation_followed_indices[condition]

        control_test[condition][layer] = evaluate_probe(
            control_probe,
            condition_states[test_indices_followed],
            correct_labels[test_indices_followed]
        )

        control_validation[condition][layer] = evaluate_probe(
            control_probe,
            condition_states[validation_indices_followed],
            correct_labels[validation_indices_followed]
        )

    print("control layer", layer, flush=True)

with (RESULTS_DIR / "control_by_layer.json").open("w", encoding="utf-8") as file:
    json.dump(
        {"test": control_test, "validation": control_validation},
        file,
        indent=2
    )

# Select one control layer using validation accuracy.
mean_validation_accuracy = {
    layer: float(np.mean([
        control_validation[condition][layer]
        for condition in CONDITIONS
    ]))
    for layer in layers
}

selected_control_layer = max(
    mean_validation_accuracy,
    key=mean_validation_accuracy.get
)

print(
    f"\nSelected control layer: {selected_control_layer} "
    f"(mean validation accuracy: {mean_validation_accuracy[selected_control_layer]:.1%})"
)

# Report held-out test results.
print("\nControl test accuracy at selected layer (hint-exclusion baseline = 33.3%)")

for condition in CONDITIONS:
    evaluation_count = len(followed_test_indices[condition])
    probe_accuracy = control_test[condition][selected_control_layer]

    correct_prediction_count = round(probe_accuracy * evaluation_count)

    confidence_interval = binomtest(
        correct_prediction_count,
        evaluation_count
    ).proportion_ci(
        confidence_level=0.95,
        method="wilson"
    )

    print(
        f"  {condition}: {probe_accuracy:.1%} "
        f"(95% CI: {confidence_interval.low:.1%}–"
        f"{confidence_interval.high:.1%}, n={evaluation_count})"
    )

# Plot control performance across layers.
fig, ax = plt.subplots(figsize=(8, 5))

for condition in CONDITIONS:
    accuracies = [
        control_test[condition][layer]
        for layer in layers
    ]
    ax.plot(layers, accuracies, label=condition)

ax.axhline(
    1 / 3,
    color="orange",
    linestyle=":",
    label="Hint-exclusion baseline (33.3%)"
)

ax.axvline(
    selected_control_layer,
    color="gray",
    linestyle="--",
    label="Validation-selected layer"
)

ax.set_xlabel("Layer")
ax.set_ylabel("Accuracy on followed-suggestion test questions")
ax.set_title("Probe trained only on pressure-resistant questions")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "control_by_layer.png", dpi=150)
plt.close(fig)
