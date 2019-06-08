import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nets import MusicRNN
from providers import MIDIConverter, MIDIDataset
from utils import generate_sample


def train(model, provider, optimizer, criterion, iterations, device, model_comb, scheduler=None):
    mean_losses = []
    losses = []
    try:
        for iteration in tqdm(range(iterations)):
            if not scheduler is None:
                scheduler.step()
            data, targets = provider.get_batch()
            data, targets = torch.tensor(data=data, dtype=torch.long).to(device), torch.tensor(data=targets, dtype=torch.long).to(device)
            model.zero_grad()

            target_preds = F.log_softmax(model(data), dim=1)

            loss = criterion(target_preds, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())
            if len(losses) == 1000:
                mean_losses.append(np.mean(losses))
                losses = []
        
            if iteration % 100000 == 0 or iteration == iterations - 1:
                with torch.no_grad():
                    sample = generate_sample(model, provider, device=device, length=100, temperature=1.0)
                    midi = provider.to_midi(sample)
                    midi.write(f'../music/{model_comb}/sample_{iteration}.midi')
    except KeyboardInterrupt:
        pass

    return mean_losses



if __name__ == '__main__':

    #### MODIFY #####
    dataset='chopin'
    dataset_path = f'/home/bartek/Datasets/{dataset}/'
    fs = 5
    seq_len = 50
    embed_size = 128
    rnn_type = 'LSTM'
    #################


    model_comb = f'{rnn_type.lower()}3_{embed_size}_{seq_len}_sgd_{dataset}'
    converter = MIDIConverter(dataset_path, frac=1.0, fs=fs)
    provider = MIDIDataset(converter, nn_batch_size=64, song_batch_size=8, seq_len=seq_len)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = MusicRNN(embedding_dim=embed_size, hidden_dim=256, unique_notes=provider.unique_notes(), seq_len=seq_len, rnn=rnn_type)
    model = model.to(device)
    loss = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300000)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = train(model, provider, optimizer, loss, 900000, device, model_comb, scheduler)
    
    torch.save(model.state_dict(), f'../models/{model_comb}/model.pb')

    fig = plt.figure()
    plt.plot([i for i in range(len(losses))], losses)
    plt.yscale('log')
    plt.savefig(f'../models/{model_comb}/losses.png')