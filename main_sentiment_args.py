import torch
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from GloveEmbed import _get_embedding
import time
from LSTM import LSTMModel
import os 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import argparse

'''save checkpoint'''
def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
    os.makedirs(os.path.dirname(ckp_path), exist_ok=True)
    checkpoint = {
        'epoch': epoches,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, ckp_path)

'''save results'''
def save_results(cm, acc, precision, recall, f1, model_config, file_path="results.txt"):
    with open(file_path, 'w') as f:
        f.write("===== Confusion Matrix =====\n")
        f.write(np.array2string(cm))
        f.write("\n\n===== Evaluation Metrics =====\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write("\n===== Model Hyperparameters =====\n")
        for k, v in model_config.items():
            f.write(f"{k}: {v}\n")
    print("Saved results to", file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--glove_file', type=str, default='glove.6B/glove.6B.300d.txt')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    gpu_id = 0
    device = torch.device("cuda", gpu_id) if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)

    # Hyperparameters
    mode = 'train'
    Batch_size = 300
    n_layers = 1
    input_len = 150
    embedding_dim = args.embedding_dim
    hidden_dim = 50
    output_size = 1
    num_epoches = args.epochs
    learning_rate = 0.002
    clip = 5
    load_cpt = False
    ckp_path = f'cpt/model_{embedding_dim}{"_ft" if args.fine_tune else "_frozen" if args.pretrain else "_default"}.pt'

    # Load data
    training_set = MovieDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size, shuffle=True, num_workers=1)
    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size, shuffle=False, num_workers=1)

    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if args.pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(args.glove_file, tokens2index, embedding_dim)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
    else:
        embedding_matrix = None

    model = LSTMModel(
        vocab_size=vocab_size,
        output_size=output_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        input_len=input_len,
        pretrain=args.pretrain
    )

    model = model.to(device)

    # Optionally fine-tune embedding
    if args.pretrain:
        model.embedding.weight.requires_grad = args.fine_tune

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = nn.BCELoss()

    if load_cpt:
        print("*" * 10 + 'loading checkpoint' + "*" * 10)
        checkpoint = torch.load(ckp_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print('*' * 89)
    print('start model training now')
    print('*' * 89)

    if mode == 'train':
        model.train()
        global_step = 0
        losses = []
        for epoch in range(num_epoches):
            for x_batch, y_labels in training_generator:
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                y_out = model(x_batch)
                loss = loss_fun(y_out, y_labels)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                global_step += 1

            print("**** save checkpoint ****")
            _save_checkpoint(ckp_path, model, epoch, global_step, optimizer)

    # Plot and save loss curve
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")

    print("----model testing now----")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_labels in test_generator:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_out = model(x_batch)
            y_pred = torch.round(y_out)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")

    model_config = {
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'batch_size': Batch_size,
        'input_len': input_len,
        'pretrain': args.pretrain,
        'fine_tune': args.fine_tune,
        'glove_file': args.glove_file if args.pretrain else 'None',
        'num_epoches': num_epoches,
        'learning_rate': learning_rate
    }

    filename = f"results_{embedding_dim}"
    if args.pretrain:
        filename += "_glove"
        filename += "_finetune" if args.fine_tune else "_frozen"
    else:
        filename += "_default"
    filename += ".txt"

    save_results(cm, acc, precision, recall, f1, model_config, filename)

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
