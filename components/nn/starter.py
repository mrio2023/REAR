import random
import torch
import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple, Set, Any

from .graph import Graph
from .Agent import Agent
from .expander import Expander
from .tool import compute_single_metrics, safe_mean
from .dataProcess import DataLoader
from .tee import Tee
from .refiner import Refiner


class Starter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.params.setdefault("max_iter", 1)

    def eval_model(self, expander: Expander, test_g: Graph, refiner=None) -> None:
        """Evaluate model: compute original metrics and optionally refined metrics."""
        expander.model.eval()

        # Prepare data
        true_coms: List[Tuple[str, List]] = [
            (tag, list(addr_set))
            for tag, addr_set in test_g.communities.items()
            if isinstance(addr_set, (set, list)) and len(addr_set) > 0
        ]
        print(f"Number of communities: {len(true_coms)}")

        test_seeds: Dict[str, List] = {}
        for name_tag, addr_list in true_coms:
            test_seeds[name_tag] = random.sample(addr_list, k=1)

        print(f"Test: {len(true_coms)} true communities, total seeds: {sum(len(v) for v in test_seeds.values())}")
        print("-" * 60)

        # Initial inference
        all_seeds_flat = []
        community_map = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))

        pred_coms_flat = []
        if all_seeds_flat:
            with torch.no_grad():
                pred_coms_flat, _ = expander.sample_bs_trajectories(all_seeds_flat)

        # Original predictions (without refinement)
        community_pred_original: Dict[str, Set] = {tag: set() for tag in test_seeds.keys()}
        for idx, pred_com in enumerate(pred_coms_flat):
            if idx < len(community_map):
                valid_nodes = [n for n in pred_com if n != "Stp"]
                community_pred_original[community_map[idx]].update(valid_nodes)

        # Compute original metrics
        metrics_original = {"precision": [], "recall": [], "f1": [], "jaccard": []}
        for name_tag, true_addr in true_coms:
            pred_com = list(community_pred_original.get(name_tag, set()))
            compute_single_metrics(pred_com, true_addr, metrics_original)

        avg_p_orig = safe_mean(metrics_original["precision"])
        avg_r_orig = safe_mean(metrics_original["recall"])
        avg_f1_orig = safe_mean(metrics_original["f1"])
        avg_j_orig = safe_mean(metrics_original["jaccard"])

        print("\n" + "=" * 80)
        print("Original Evaluation Metrics (without refinement)")
        print("=" * 80)
        print(f"Precision: {avg_p_orig:.4f} | Recall: {avg_r_orig:.4f} | F1: {avg_f1_orig:.4f} | Jaccard: {avg_j_orig:.4f}")

        # If refiner is provided, refine and compute refined metrics
        if refiner is not None:
            community_pred_refined = {tag: set(community_pred_original[tag]) for tag in community_pred_original.keys()}
            for tag in community_pred_refined.keys():
                comm_list = list(community_pred_refined[tag])
                refined = refiner.refine_community(comm_list, threshold=0.3)
                community_pred_refined[tag] = set(refined)
            print("Refiner applied (noise removal)")

            metrics_refined = {"precision": [], "recall": [], "f1": [], "jaccard": []}
            for name_tag, true_addr in true_coms:
                pred_com = list(community_pred_refined.get(name_tag, set()))
                compute_single_metrics(pred_com, true_addr, metrics_refined)

            avg_p_ref = safe_mean(metrics_refined["precision"])
            avg_r_ref = safe_mean(metrics_refined["recall"])
            avg_f1_ref = safe_mean(metrics_refined["f1"])
            avg_j_ref = safe_mean(metrics_refined["jaccard"])

            print("\n" + "=" * 80)
            print("Refined Evaluation Metrics")
            print("=" * 80)
            print(f"Precision: {avg_p_ref:.4f} | Recall: {avg_r_ref:.4f} | F1: {avg_f1_ref:.4f} | Jaccard: {avg_j_ref:.4f}")
            print("\n" + "=" * 80)
            print("Metric Changes")
            print("=" * 80)
            print(f"Precision: {avg_p_orig:.4f} -> {avg_p_ref:.4f} ({avg_p_ref - avg_p_orig:+.4f})")
            print(f"Recall:    {avg_r_orig:.4f} -> {avg_r_ref:.4f} ({avg_r_ref - avg_r_orig:+.4f})")
            print(f"F1:        {avg_f1_orig:.4f} -> {avg_f1_ref:.4f} ({avg_f1_ref - avg_f1_orig:+.4f})")
            print(f"Jaccard:   {avg_j_orig:.4f} -> {avg_j_ref:.4f} ({avg_j_ref - avg_j_orig:+.4f})")
            print("=" * 80)

    def run(self, dfname: str, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.params["device"] == "cuda":
            torch.cuda.manual_seed(seed)

        log_dir = os.path.join("logs", dfname)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{dfname}_seed{seed}_{timestamp}.log")

        f = None
        original_stdout = sys.stdout
        tee = None

        try:
            f = open(log_file, "w", encoding="utf-8")
            tee = Tee(f, original_stdout)
            sys.stdout = tee

            print(f"Dataset: {dfname}  seed={seed}")
            print(f"Parameters: min_community_size={self.params['min_community_size']}, epoch={self.params['epoch']}, seedNum={self.params['seedNum']}")
            print("-" * 70)

            loader = DataLoader(
                dataset_name=dfname,
                train_ratio=0.8,
                min_com_size=self.params["min_community_size"],
                seed=seed,
            )

            global_adj = loader.graph["adj"]
            global_features = loader.nodefeats

            train_communities = {f"train_{idx}": comm for idx, comm in enumerate(loader.train_comms)}
            test_communities = {f"test_{idx}": comm for idx, comm in enumerate(loader.test_comms)}

            train_g = Graph(
                adj=global_adj, features=global_features, communities=train_communities
            )
            test_g = Graph(
                adj=global_adj, features=global_features, communities=test_communities
            )

            print(f"Train graph: nodes={train_g.n_nodes}, embedding_dim={train_g.embedsize}, communities={len(train_communities)}")
            print(f"Test graph: nodes={test_g.n_nodes}, embedding_dim={test_g.embedsize}, communities={len(test_communities)}")

            device = torch.device(self.params["device"])
            model = Agent(
                input_size=train_g.embedsize, hidden_size=self.params["hidden_size"]
            ).to(device)

            expander = Expander(
                graph=train_g,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=self.params["lr"]),
                device=device,
                maxLen=self.params["maxLen"],
                gamma=self.params["gamma"],
                f1_base_weight=self.params["f1_base_weight"],
                p_bias=self.params["p_bias"],
                r_bias=self.params["r_bias"],
                grad_norm=self.params["grad_norm"],
                target_f1=self.params["target_f1"],
                stop_reward_scale=self.params["stop_reward_scale"],
            )

            epoch = self.params["epoch"]
            seedNum = self.params["seedNum"]
            coms = list(train_g.communities.values())

            print(f"Training: {epoch} epochs, {seedNum} seeds per epoch")
            for i in range(epoch):
                true_coms = random.sample(coms, k=seedNum)
                seeds = [random.choice(c) for c in true_coms]
                loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
                print(f"Epoch {i+1}/{epoch} | loss: {loss:.4f}")

            print("\nTraining Refiner...")
            refiner = Refiner(train_g, expander)
            refiner.trainRefiner()

            print("\nTesting (true communities + dynamic seed sampling)")
            expander.graph = test_g
            refiner.train_g = test_g
            self.eval_model(expander, test_g, refiner=refiner)

        except Exception as e:
            print(f"\nRuntime error: {str(e)}", file=sys.stderr)
            raise
        finally:
            sys.stdout = original_stdout
            if tee:
                tee.flush()
            if f and not f.closed:
                f.close()
            print(f"\nLog saved to: {log_file}")