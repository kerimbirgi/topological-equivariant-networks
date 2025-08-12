import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import Accuracy, MeanAbsoluteError, R2Score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_paths(config):
    if not os.path.isdir(config['path']['predictions']):
        os.makedirs(config['path']['predictions'], exist_ok=True)
    if not os.path.isdir(config['path']['plots']):
        os.makedirs(config['path']['plots'], exist_ok=True)
    return

def save_predictions(ids, predictions, targets, config):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    df = pd.DataFrame({'protein_ligand_id': ids,'predictions': predictions, 'targets': targets})
    df.to_csv(config['path']['predictions'] / 'predictions.csv', index=False)
    return predictions, targets

def read_predictions(path):
    df = pd.read_csv(path)
    ids = df['protein_ligand_id'].values
    predictions = df['predictions'].values
    targets = df['targets'].values
    return ids, predictions, targets

def plot_classification(predictions, targets, log_path=None):
    pred_counts_false = np.sum(predictions == 0)
    targ_counts_false = np.sum(targets == 0)
    pred_counts_true = np.sum(predictions == 1)
    targ_counts_true = np.sum(targets == 1)
    pred_counts = [pred_counts_false, pred_counts_true]
    targ_counts = [targ_counts_false, targ_counts_true]
    classes = ['False', 'True']
    bar_width = 0.35
    x = np.arange(len(classes))
    plt.figure(figsize=(6, 4))
    plt.bar(x - bar_width/2, pred_counts, width=bar_width, label='Predictions')
    plt.bar(x + bar_width/2, targ_counts, width=bar_width, label='Targets')
    plt.xticks(x, classes)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Predicted vs True Class Distribution')
    plt.legend()
    plt.tight_layout()
    #plt.savefig("/Users/kerimbirgi/dev/bindingnet_research/outputs/plots/dual_transformer_classification_subset_20p_with_false/predictions_vs_targets_classification.png")
    plt.close()
    # boolean masks
    tp_mask = (predictions == 1) & (targets == 1)
    tn_mask = (predictions == 0) & (targets == 0)
    fp_mask = (predictions == 1) & (targets == 0)
    fn_mask = (predictions == 0) & (targets == 1)

    # counts
    tp = tp_mask.sum().item()
    tn = tn_mask.sum().item()
    fp = fp_mask.sum().item()
    fn = fn_mask.sum().item()

    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("tn:", tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy: {accuracy}")
    #accuracy = (predictions == targets).mean()
    #print(f"Accuracy: {accuracy}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    print(f"Precision: {precision}")
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"Recall: {recall}")
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"F1 Score: {f1_score}")

    AUC = roc_auc_score(targets, predictions)
    print(f"AUC: {AUC}")
    # plot roc curve
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'AUC = {AUC:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("/Users/kerimbirgi/dev/bindingnet_research/outputs/plots/dual_transformer_classification_subset_20p_with_false/roc_curve_classification.png")
    plt.close()
    return accuracy, precision, recall, f1_score, AUC

def plot_regression(predictions, targets, log_path=None):
    assert len(predictions) == len(targets)
    # Calculate regression losses/metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    log_file = open(log_path / "results.txt", 'w') if log_path else None
    def log(*args, **kwargs):
        print(*args, **kwargs, file=log_file if log_file else None)

    log("Predictions range:", predictions.min(), predictions.max())
    log("Targets range:", targets.min(), targets.max())
    log("Mean of targets:", targets.mean())
    log("Mean of predictions:", predictions.mean())
    log(f"MSE: {mse}")
    log(f"RMSE: {rmse}")
    log(f"MAE: {mae}")

    lo = min(predictions.min(), targets.min())
    hi = max(predictions.max(), targets.max())
    bins = np.linspace(lo, hi, 51)        # 50 equal-width bins
    plt.figure(figsize=(10, 5))
    plt.hist(predictions, bins=bins, alpha=0.5, label='Predictions')
    plt.hist(targets, bins=bins, alpha=0.5, label='Targets')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    plt.legend()
    plt.savefig(log_path / "predictions_distribution_regression.png")
    plt.close()

def create_plots(ids, predictions, targets, config):
    if config['task'] == 'classification':
        plot_classification(predictions, targets, log_path=config['path']['plots'] / "results.txt")
    elif config['task'] == 'regression':
        plot_regression(predictions, targets, log_path=config['path']['plots'])

def get_predictions_doubleGine3D(model, test_loader, device):
    predictions = []
    targets = []
    ids = []
    with torch.no_grad():
        for protein_ligand_id_batch, protein_graph_batch, ligand_graph_batch, labels_batch in tqdm(test_loader, total=len(test_loader), desc="Evaluating"):
            protein_graph_batch = protein_graph_batch.to(device)
            ligand_graph_batch = ligand_graph_batch.to(device)
            #labels_batch = to_device(labels_batch, device)

            #outputs = model(protein_graph_batch, ligand_graph_batch)
            outputs = model(protein_graph_batch, ligand_graph_batch).squeeze(-1)
            if outputs.ndim != 1:
                raise Exception("Shape of output not correct")
            
            predictions.append(outputs.cpu())
            targets.append(labels_batch)
            ids.extend(protein_ligand_id_batch)
            
    #predictions = torch.cat(predictions, dim=0)
    #targets = torch.cat(targets, dim=0)
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    return ids, predictions, targets


def evaluate(config, model, test_dataset):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("device is cuda")
        print("Cuda available?", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("device is mps")
    else:
        raise Exception("Neither cuda nor mps is available.")

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['evaluate']['batch_size'], 
        pin_memory=config['evaluate']['pin_memory'], 
        num_workers=config['evaluate']['num_workers'],
        shuffle=False,
        collate_fn=test_dataset.pyg_collate
    )

    model = model.to(device)
    model.eval()
    print("Data loaders created!")
    print("Evaluating model...")
    ids, predictions, targets = get_predictions_doubleGine3D(model, test_loader, device)

    if config['evaluate']['save_predictions']:
        predictions, targets = save_predictions(ids, predictions, targets, config)
    print("Predictions saved!")
    
    if config['evaluate']['create_plots']:
        create_plots(ids, predictions, targets, config)
    print("Plots created!")

    
    
        

def main(config):
    ensure_paths(config)

    print("Loading data...")
    test_df = pd.read_csv(str(config['path']['data']))
    test_indices = np.load(str(config['path']['indices'] / 'test_indices.npy'))
    print("Data loaded!")
    print(f"Test Samples: {len(test_indices)}")

    test_dataset = doubleGine3DDataset(
            test_df, 
            indices=test_indices, 
            protein_graph_path=config['path']['protein_graphs'],
            ligand_graph_path=config['path']['ligand_graphs']
            )

    print("Creating model...")
    model = load_model(config)
    print("Model created!")

    print("Evaluating model...")
    evaluate(config, model, test_dataset)
    print("Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--no_eval', action="store_true", help="If this flag is set predictions are not run and read instead.")
    args = parser.parse_args()
    try:

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config['path'].items(): # convert to Path objects
            config['path'][key] = Path(value)
    except Exception as e:
        print(f"Error loading config file: {args.config} - {e}")
        exit(1)
        
    if args.no_eval:
        print("Evaluating and plotting existing predictions!")
        ids, predictions, targets = read_predictions(config['path']['predictions'] / 'predictions.csv')
        create_plots(ids, predictions, targets, config)
        print("Evaluation and plotting complete!")
    else:
        main(config)
    print("OK!")
