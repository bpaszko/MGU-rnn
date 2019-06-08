import numpy as np
import torch
import torch.nn.functional as F
from collections import deque


def generate_sample(model, provider, length=50, temperature=1.0, device='cpu'):
    notes = deque(provider.sample_start(1))
    generated = [notes[-1]]
    for i in range(length-1):
        data = torch.tensor(data=np.expand_dims(np.array(notes), axis=0), dtype=torch.long).to(device)
        output = F.softmax(model(data) / temperature, dim=1).cpu()
        output = torch.clamp(output, min=0)
        new_note = torch.multinomial(output, 1)[:, 0]
        new_note = new_note.item()
        notes.popleft()
        notes.append(new_note)
        generated.append(new_note)
    return generated