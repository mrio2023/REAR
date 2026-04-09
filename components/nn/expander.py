from typing import Optional, List
import numpy as np
import torch
from .tool import eval_scores, pruning


class Expander:
    def __init__(
        self,
        graph,
        model,
        maxLen: int,
        optimizer,
        device: Optional[torch.device],
        gamma: float,
        f1_base_weight: float,
        p_bias: float,
        r_bias: float,
        grad_norm: float,
        target_f1: float,
        stop_reward_scale: float,
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.device = device

        self.f1_base_weight = f1_base_weight
        self.p_bias = p_bias
        self.r_bias = r_bias
        self.target_f1 = target_f1
        self.grad_norm = grad_norm
        self.stop_reward_scale = stop_reward_scale

        print(f"Expander initialized: maxLen={maxLen}, gamma={gamma}, "
              f"f1_base_weight={f1_base_weight}, device={self.device}")
        print(f"p_bias={p_bias}, r_bias={r_bias}, grad_norm={grad_norm}")
        print(f"target_f1={target_f1}, stop_reward_scale={stop_reward_scale}")

    def sample_actions(self, logits, training=True):
        """
        Sample actions according to mode:
        - Training: sample stochastically
        - Evaluation: greedy selection
        """
        actions, log_probs = [], []

        for i, batch_logits in enumerate(logits):
            if batch_logits is None or batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(torch.tensor(0.0, device=self.device))
                continue

            # Sanity: clamp extreme values
            batch_logits = torch.clamp(batch_logits, min=-20, max=20)
            dist = torch.distributions.Categorical(logits=batch_logits)

            if training:
                action = dist.sample()
            else:
                action = torch.argmax(dist.probs)

            log_prob = dist.log_prob(action)

            # Last action is assumed to be the stop action
            if action == len(dist.probs) - 1:
                actions.append("Stp")
            else:
                actions.append(action.item())

            log_probs.append(log_prob)

        return actions, log_probs

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        """
        Prepare model inputs: seed embeddings, trajectory node embeddings,
        indptr, and candidate node lists.
        """
        t_tra_vector, t_seed_vector, indptr, choices = [], [], [], []
        offset = 0
        embed_dim = self.graph.embedsize

        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))

            # No neighbours: only stop action
            if len(unique_neigh) == 0:
                t_tra_vector.append(tra_vector[i].unsqueeze(0))
                t_seed_vector.append(seed_vector[i].unsqueeze(0))
                choices.append([])
                indptr.append((offset, offset + 1, offset))
                offset += 1
                continue

            # Normal case: prune neighbours
            neigh_embed = self.graph.nodesEmbed(unique_neigh)

            pruned_nodes = pruning(
                community_pooled_embed=tra_vector[i],
                neigh_node_embed_list=neigh_embed,
                neigh_nodes=unique_neigh,
            )

            idx_map = {node: j for j, node in enumerate(unique_neigh)}
            keep_idx = [idx_map[n] for n in pruned_nodes]

            if isinstance(neigh_embed, list):
                neigh_embed = [neigh_embed[j] for j in keep_idx]
            else:
                neigh_embed = neigh_embed[keep_idx]
            unique_neigh = pruned_nodes

            # Convert to tensor
            if not isinstance(neigh_embed, torch.Tensor):
                neigh_embed = torch.tensor(
                    np.stack(neigh_embed) if isinstance(neigh_embed, list) else neigh_embed,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                neigh_embed = neigh_embed.to(self.device)

            # Add neighbour embeddings
            for n in neigh_embed:
                t_tra_vector.append(n.unsqueeze(0))
                t_seed_vector.append(n.unsqueeze(0))
            # Add current node embedding
            t_tra_vector.append(tra_vector[i].unsqueeze(0))
            t_seed_vector.append(seed_vector[i].unsqueeze(0))

            choices.append(unique_neigh)
            indptr.append(
                (offset, offset + 1 + len(unique_neigh), offset + len(unique_neigh))
            )
            offset += len(unique_neigh) + 1

        if not t_seed_vector:
            seed_embed = torch.empty((0, embed_dim), device=self.device)
            tra_embed = torch.empty((0, embed_dim), device=self.device)
        else:
            seed_embed = torch.cat(t_seed_vector, dim=0)
            tra_embed = torch.cat(t_tra_vector, dim=0)

        return seed_embed, tra_embed, np.array(indptr), choices

    def add_node(self, new_node, tra_nodes, tra_sets, index):
        if new_node in (None, "Stp", -1) or len(tra_nodes[index]) >= self.maxLen:
            if new_node == "Stp":
                tra_nodes[index].append("Stp")
            self.done[index] = True
            return None
        if new_node in tra_sets[index]:
            return None
        tra_nodes[index].append(new_node)
        tra_sets[index].add(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        embed = (
            torch.tensor(embed, dtype=torch.float32, device=self.device)
            if not isinstance(embed, torch.Tensor)
            else embed.to(self.device)
        )
        return embed

    def all_done(self):
        return all(self.done) if hasattr(self, "done") else False

    def vecpool(self, v1, v2, k):
        return (v1 * (k - 1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        """Sample a batch of trajectories (for evaluation)."""
        seed_vector = self.graph.nodesEmbed(seeds)
        seed_vector = (
            torch.tensor(
                np.stack(seed_vector) if isinstance(seed_vector, list) else seed_vector,
                dtype=torch.float32,
                device=self.device,
            )
            if not isinstance(seed_vector, torch.Tensor)
            else seed_vector.to(self.device)
        )

        tra_vector = seed_vector.clone()
        tra_nodes = [[s] for s in seeds]
        tra_sets = [{s} for s in seeds]
        tra_logps = [[] for _ in range(len(seeds))]

        self.done = [False] * len(seeds)
        step = 0

        while step < self.maxLen and not self.all_done():
            active_indices = [i for i, d in enumerate(self.done) if not d]
            if not active_indices:
                break

            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]

            *model_inputs, batch_candidates = self.prepare_inputs(
                active_tra_vector, active_seed_vector, active_tra_nodes
            )
            batch_logits = self.model(*model_inputs)

            actions, logps = self.sample_actions(batch_logits, training=False)

            for j, orig_idx in enumerate(active_indices):
                ac = actions[j]
                logp = logps[j]

                if batch_candidates[j] == []:
                    self.add_node("Stp", tra_nodes, tra_sets, orig_idx)
                    tra_logps[orig_idx].append(logp)
                    continue

                if ac == "Stp" or ac >= len(batch_candidates[j]):
                    self.add_node("Stp", tra_nodes, tra_sets, orig_idx)
                    tra_logps[orig_idx].append(logp)
                else:
                    selected_node = batch_candidates[j][ac]
                    newvec = self.add_node(selected_node, tra_nodes, tra_sets, orig_idx)
                    if newvec is not None:
                        tra_vector[orig_idx] = self.vecpool(
                            tra_vector[orig_idx], newvec, len(tra_nodes[orig_idx])
                        )
                    tra_logps[orig_idx].append(logp)

            step += 1

        return tra_nodes, tra_logps

    def trainReward(self, seeds: List[int], true_coms):
        self.model.train()

        selected_nodes, logps = self.sample_bs_trajectories(seeds)

        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        p_list, r_list, f1_list, pred_len_list = [], [], [], []
        len_true_sum = 0
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            len_true_sum += len(true_com)
            p, r, f1 = eval_scores(pred_com_clean, true_com)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            pred_len_list.append(len(pred_com_clean))

        print(f"Average true community length: {len_true_sum / len(true_coms):.3f}")
        batch_recall = np.mean(r_list)
        batch_precision = np.mean(p_list)
        batch_f1 = np.mean(f1_list)
        avg_ext_len = np.mean(pred_len_list)

        print(
            f"Batch Metrics: P={batch_precision:.4f}, R={batch_recall:.4f}, "
            f"F1={batch_f1:.4f}, AvgExtLen={avg_ext_len:.2f}"
        )

        # Compute rewards
        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com = [com[0]]
            temp_set = {com[0]}
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)

            step_rewards = []

            for node in com[1:]:
                if node == "Stp":
                    intersect = len(temp_set & true_com_set)
                    curr_p = intersect / len(temp_set) if len(temp_set) > 0 else 0.0
                    curr_r = intersect / true_com_len
                    if curr_p + curr_r > 0:
                        curr_f1 = 2 * curr_p * curr_r / (curr_p + curr_r)
                    else:
                        curr_f1 = 0.0
                    stop_reward = (curr_f1 - self.target_f1) * self.stop_reward_scale
                    step_rewards.append(stop_reward)
                    continue

                pre_intersect = len(temp_set & true_com_set)
                pre_p = pre_intersect / len(temp_set) if temp_set else 0.0
                pre_r = pre_intersect / true_com_len

                temp_com.append(node)
                temp_set.add(node)

                curr_intersect = len(temp_set & true_com_set)
                curr_p = curr_intersect / len(temp_set)
                curr_r = curr_intersect / true_com_len

                recall_inc = curr_r - pre_r
                precision_inc = curr_p - pre_p

                base_reward = (
                    recall_inc * self.r_bias + precision_inc * self.p_bias
                ) * self.f1_base_weight
                step_rewards.append(base_reward)

            # Discounted returns
            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            else:
                discounted = [0.0]
            rewards_list.append(discounted)

        # Pad to same length
        max_len_pad = max(
            max(len(r) for r in rewards_list), max(len(lp) for lp in logps)
        )
        rewards_padded = np.zeros((bs, max_len_pad))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, : len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        mask = (
            torch.arange(max_len_pad, device=self.device).expand(bs, -1)
            < (lengths - 1).unsqueeze(1)
        ).float()

        # Build log_probs tensor
        logps_padded = []
        for lp_list in logps:
            padded = [
                lp_list[j] if j < len(lp_list) else torch.tensor(0.0, device=self.device)
                for j in range(max_len_pad)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        pg_loss = -(rewards.detach() * logps * mask).sum()
        loss = pg_loss

        print(f"Loss | PG: {pg_loss.item():.4f}, Total: {loss.item():.4f}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_val = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_norm
        )
        self.optimizer.step()

        print(f"Gradient norm after clipping: {grad_norm_val:.4f}")

        return loss.item()