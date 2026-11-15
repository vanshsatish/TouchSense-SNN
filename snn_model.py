from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

try:
    import snntorch as snn
except ImportError:
    snn = None

class TouchSNN(nn.Module):
    """Small recurrent-in-time LIF network over event-coded feature sequences."""
    def __init__(self, input_size: int = 5, hidden_size: int = 32, classes: int = 3):
        super().__init__()
        if snn is None:
            raise RuntimeError("snntorch is not installed. Run pip install -r requirements.txt")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(hidden_size, classes)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, sequence: torch.Tensor):
        # sequence shape: [batch, time_steps, input_size], values in [0, 1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        outputs = []
        for t in range(sequence.shape[1]):
            current1 = self.fc1(sequence[:, t, :])
            spk1, mem1 = self.lif1(current1, mem1)
            current2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(current2, mem2)
            outputs.append(spk2)
        return torch.stack(outputs, dim=1), mem2

def load_checkpoint(path: str | Path, device="cpu"):
    path = Path(path)
    if not path.exists():
        return None, "not_trained"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TouchSNN(
        input_size=checkpoint.get("input_size", 5),
        hidden_size=checkpoint.get("hidden_size", 32),
        classes=checkpoint.get("classes", 3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, "trained"
