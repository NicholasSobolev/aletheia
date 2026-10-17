import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from experiment import DEVICE, build_prompt, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATES_DIR = PROJECT_ROOT / "states"

with (DATA_DIR / "flip_results.json").open("r", encoding="utf-8") as file:
    items = json.load(file)

tokenizer, model = load_model()


@torch.no_grad()
def extract_states(
    prompts: list[str], batch_size: int = 8
) -> NDArray[np.float16]:
    """Extract final-token hidden states from every model layer."""
    hidden_states = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)

        layers = model(**encoded, output_hidden_states=True).hidden_states

        final_token_states = torch.stack(
            [layer[:, -1, :] for layer in layers],
            dim=1,
        )  # (batch, layers + 1, hidden_size)

        hidden_states.append(
            final_token_states.float().cpu().numpy().astype(np.float16)
        )

    return np.concatenate(hidden_states)


CONDITIONS = [
    "neutral",
    "user_belief_wrong",
    "textbook_wrong",
    "answer_key_wrong",
    "unattributed_wrong",
]

STATES_DIR.mkdir(exist_ok=True)

for condition in CONDITIONS:
    prompts = [build_prompt(item, condition, tokenizer) for item in items]
    activations = extract_states(prompts)

    np.save(STATES_DIR / f"{condition}.npy", activations)
    print(condition, activations.shape, flush=True)
