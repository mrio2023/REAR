import sys
import os
import numpy as np
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch import optim

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from codes.components.nn.dataProcess import DataProcess
from codes.components.nn.ellipticDataProcess import ellipticDataProcess
from codes.components.nn.ellipticGraph import ellipticGraph
from codes.components.nn.graph import Graph
from codes.components.nn.Agent import Agent
from codes.components.nn.expander import Expander

def plot_training_loss(history, dfname, save_dir="./"):
    save_path = os.path.join(save_dir, f"{dfname}_training_loss.png")
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training loss')
    plt.axhline(y=np.mean(history['loss']), color='r', linestyle='--',
                label=f'Mean loss: {np.mean(history["loss"]):.4f}')
    plt.title(f'{dfname} Training Loss Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_test_metrics(pred_results, dfname, save_dir="./"):
    save_path = os.path.join(save_dir, f"{dfname}_test_metrics.png")
    seeds = [f"Seed {i+1}" for i in range(len(pred_results['seed_node']))]
    precision = pred_results['precision']
    recall = pred_results['recall']
    f1 = pred_results['f1']
    jaccard = pred_results['jaccard']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dfname} Test Metrics', fontsize=16)

    axes[0,0].bar(seeds, precision, color='skyblue', alpha=0.8)
    axes[0,0].axhline(y=np.mean(precision), color='r', linestyle='--',
                      label=f'Mean: {np.mean(precision):.4f}')
    axes[0,0].set_title('Precision', fontsize=12)
    axes[0,0].set_ylabel('Value', fontsize=10)
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    axes[0,1].bar(seeds, recall, color='lightgreen', alpha=0.8)
    axes[0,1].axhline(y=np.mean(recall), color='r', linestyle='--',
                      label=f'Mean: {np.mean(recall):.4f}')
    axes[0,1].set_title('Recall', fontsize=12)
    axes[0,1].set_ylabel('Value', fontsize=10)
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    axes[1,0].bar(seeds, f1, color='orange', alpha=0.8)
    axes[1,0].axhline(y=np.mean(f1), color='r', linestyle='--',
                      label=f'Mean: {np.mean(f1):.4f}')
    axes[1,0].set_title('F1-Score', fontsize=12)
    axes[1,0].set_xlabel('Seed Nodes', fontsize=10)
    axes[1,0].set_ylabel('Value', fontsize=10)
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    axes[1,1].bar(seeds, jaccard, color='purple', alpha=0.8)
    axes[1,1].axhline(y=np.mean(jaccard), color='r', linestyle='--',
                      label=f'Mean: {np.mean(jaccard):.4f}')
    axes[1,1].set_title('Jaccard Similarity', fontsize=12)
    axes[1,1].set_xlabel('Seed Nodes', fontsize=10)
    axes[1,1].set_ylabel('Value', fontsize=10)
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_community_size(pred_results, dfname, save_dir="./"):
    save_path = os.path.join(save_dir, f"{dfname}_community_sizes.png")
    seeds = [f"Seed {i+1}" for i in range(len(pred_results['seed_node']))]
    true_size = [len(c) for c in pred_results['true_community']]
    pred_size = [len(c) for c in pred_results['pred_community']]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(seeds))
    width = 0.35

    plt.bar(x - width/2, true_size, width, label='True Community Size', color='royalblue', alpha=0.8)
    plt.bar(x + width/2, pred_size, width, label='Pred Community Size', color='tomato', alpha=0.8)

    plt.title(f'{dfname} Community Size Comparison', fontsize=14)
    plt.xlabel('Test Seed Nodes', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.xticks(x, seeds)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run(dfname: str, seedNum:int, epochs: int, test_seed_num: int):
    print("=" * 60)
    print(f"Starting experiment on dataset: {dfname}")
    print(f"Epochs: {epochs} | Train seeds per epoch: {seedNum} | Test seeds: {test_seed_num}")
    print("=" * 60)

    # Train logic
    if dfname not in ["elliptic_txs", "elliptic2"]:
        d_train = DataProcess(dfname=dfname)
        train_g = Graph(
            dffeature=d_train.train_feature, 
            dfhacker=d_train.train_hacker, 
            dfnode=d_train.train_nodes
        )
    else:
        d_train = ellipticDataProcess(dfname=dfname)
        train_g = ellipticGraph(
            dfedge=d_train.train_edge, 
            dffeature=d_train.train_feature,
            dfnode=d_train.train_nodes, 
            dfhacker=d_train.train_hacker
        )

    model = Agent(input_size=train_g.embedsize, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    e_train = Expander(graph=train_g, optimizer=optimizer, model=model)
    history = {'loss': []}

    print("\n[Training Start]")
    for epoch in range(1, epochs + 1):
        train_seeds = random.sample(train_g.df_hacker["address"].values.tolist(), k=seedNum)
        isweak = []
        truecom = []

        for s in train_seeds:
            c, w = train_g.sampleTrajectory(s)
            truecom.append(c)
            isweak.append(w)

        loss = e_train.trainReward(seeds=train_seeds, true_coms=truecom, isweak=isweak)
        history['loss'].append(loss)

        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{epochs}] | Loss: {loss:.6f}")

    print("[Training Finished]")

    # ======================
    # Fixed test logic HERE
    # ======================
    if dfname not in ["elliptic_txs", "elliptic2"]:
        d_full = DataProcess(dfname=dfname)
        full_feature = pd.concat([d_full.train_feature, d_full.test_feature], ignore_index=True).drop_duplicates()
        full_nodes = pd.concat([d_full.train_nodes, d_full.test_nodes], ignore_index=True).drop_duplicates()
        full_hacker = pd.concat([d_full.train_hacker, d_full.test_hacker], ignore_index=True).drop_duplicates()
        full_g = Graph(dffeature=full_feature, dfhacker=full_hacker, dfnode=full_nodes)
        test_seeds_pool = d_full.test_hacker["address"].unique().tolist()
    else:
        d_full = ellipticDataProcess(dfname=dfname)
        full_edge = pd.concat([d_full.train_edge, d_full.test_edge], ignore_index=True).drop_duplicates()
        full_feature = pd.concat([d_full.train_feature, d_full.test_feature], ignore_index=True).drop_duplicates()
        full_nodes = pd.concat([d_full.train_nodes, d_full.test_nodes], ignore_index=True).drop_duplicates()
        full_hacker = pd.concat([d_full.train_hacker, d_full.test_hacker], ignore_index=True).drop_duplicates()
        full_g = ellipticGraph(dfedge=full_edge, dffeature=full_feature, dfnode=full_nodes, dfhacker=full_hacker)
        test_seeds_pool = d_full.test_hacker["address"].unique().tolist()

    model.eval()
    e_test = Expander(graph=full_g, optimizer=optimizer, model=model)

    if len(test_seeds_pool) < test_seed_num:
        test_seed_num = len(test_seeds_pool)
    test_seeds = random.sample(test_seeds_pool, k=test_seed_num)

    pred_results = {
        "seed_node": [],
        "pred_community": [],
        "true_community": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": []
    }

    print("\n[Testing Start]")
    with torch.no_grad():
        for idx, seed in enumerate(test_seeds):
            true_com, _ = full_g.sampleTrajectory(seed)
            true_com_set = set(true_com)

            pred_tra, _ = e_test.sample_bs_trajectories([seed])
            pred_com = [node for node in pred_tra[0] if node != "Stp" and node in full_g.deMap]
            pred_com_set = set(pred_com)

            tp = len(true_com_set & pred_com_set)
            fp = len(pred_com_set - true_com_set)
            fn = len(true_com_set - pred_com_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            jaccard = tp / (len(true_com_set | pred_com_set)) if len(true_com_set | pred_com_set) > 0 else 0.0

            pred_results["seed_node"].append(seed)
            pred_results["pred_community"].append(pred_com)
            pred_results["true_community"].append(true_com)
            pred_results["precision"].append(precision)
            pred_results["recall"].append(recall)
            pred_results["f1"].append(f1)
            pred_results["jaccard"].append(jaccard)

            print(f"Test Seed {idx+1}/{len(test_seeds)} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | Jaccard: {jaccard:.4f}")

    print("[Testing Finished]")

    print("\n[Saving plots...]")
    plot_training_loss(history, dfname)
    plot_test_metrics(pred_results, dfname)
    plot_community_size(pred_results, dfname)
    print("Plots saved successfully.\n")

    print("=" * 60)
    print(f"Experiment {dfname} completed!")
    print(f"Avg Precision: {np.mean(pred_results['precision']):.4f}")
    print(f"Avg Recall:    {np.mean(pred_results['recall']):.4f}")
    print(f"Avg F1:        {np.mean(pred_results['f1']):.4f}")
    print(f"Avg Jaccard:   {np.mean(pred_results['jaccard']):.4f}")
    print("=" * 60)

    return history, pred_results