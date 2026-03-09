def trainReward(self, seeds: List[int], true_coms):
        #这个奖励函数的recall特别高
        """核心训练逻辑：F1主导+精度倾斜的奖励函数"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 1. 计算P/R/F1指标（仅打印）
        avg_metrics = {}
        p_list, r_list, f1_list = [], [], []
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            p, r, f1 = eval_scores(pred_com_clean, true_com)
            p_list.append(p), r_list.append(r), f1_list.append(f1)

        avg_metrics = {
            "avg_precision": round(np.mean(p_list), 4),
            "avg_recall": round(np.mean(r_list), 4),
            "avg_f1": round(np.mean(f1_list), 4),
        }
        print(
            f"Batch Metrics: P={avg_metrics['avg_precision']}, R={avg_metrics['avg_recall']}, F1={avg_metrics['avg_f1']}"
        )

        # 2. F1主导的奖励计算
        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com, step_rewards = [com[0]], []
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)

            # 遍历每一步选择的节点
            for node in com[1:]:
                if node == "Stp" or node in temp_com:
                    continue

                # 计算选节点前后的F1
                pre_f1 = eval_f1(temp_com, true_com_set)
                temp_com.append(node)
                curr_f1 =eval_f1(temp_com, true_com_set)
                curr_p, curr_r, _ = eval_scores(temp_com, true_com_set)

                # 精度倾斜：P<R时放大F1奖励
                if curr_p < curr_r:
                    biased_f1 = curr_f1 * (1 + self.p_bias)
                else:
                    biased_f1 = curr_f1

                # 增量奖励：仅F1提升且高于阈值才给正向奖励
                if curr_f1 > pre_f1 and curr_f1 > self.min_f1_threshold:
                    base_reward = biased_f1 * self.f1_base_weight
                else:
                    base_reward = 0.0

                # 长度惩罚：超过真实社区长度则指数衰减
                curr_pred_len = len(temp_com)
                if curr_pred_len > true_com_len:
                    base_reward *= self.len_penalty_coeff ** (
                        curr_pred_len - true_com_len
                    )

                # 奖励约束：非负化
                base_reward = np.clip(base_reward, 0.0, self.f1_base_weight)

                step_rewards.append(base_reward)

            # 折扣奖励
            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            rewards_list.append(discounted if discounted else [0.0])

        # 3. Loss计算
        max_len = (
            max(max(len(r) for r in rewards_list), max(len(lp) for lp in logps))
            if (rewards_list and logps)
            else 0
        )
        # 填充奖励
        rewards_padded = np.zeros((bs, max_len))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, : len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        # 填充logps
        logps_padded = []
        for lp_list in logps:
            padded = [
                (
                    lp_list[j]
                    if j < len(lp_list) and lp_list[j] is not None
                    else torch.tensor(-1e9, device=self.device, requires_grad=True)
                )
                for j in range(max_len)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        # Mask与Loss计算
        mask = (
            torch.arange(max_len, device=self.device).expand(bs, -1)
            < (lengths - 1).unsqueeze(1)
        ).float()
        loss = -(rewards * logps * mask).sum()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        loss.backward()
        self.optimizer.step()

        return loss.item()



#这个奖励函数的precision特别高
def trainRewardPrecision(self, seeds: List[int], true_coms):
        """核心训练逻辑：基于batch全局指标调整最终loss（参数仅作用于batch级）"""
        self.model.train()

        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        p_list, r_list, f1_list = [], [], []
        pred_len_list = []
        early_stop_count = 0  # 新增：统计提前终止的样本数
        max_len = self.maxLen  # 取预设的最大扩展长度
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            p, r, f1 = eval_scores(pred_com_clean, true_com)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            pred_len_list.append(len(pred_com_clean))
            
            # 新增：判断是否提前终止（长度<maxLen 且 包含Stp）
            has_stp = "Stp" in pred_com
            is_early_stop = len(pred_com_clean) < max_len and has_stp
            if is_early_stop:
                early_stop_count += 1

        batch_recall = np.mean(r_list)
        batch_precision = np.mean(p_list)
        batch_f1 = np.mean(f1_list)
        avg_ext_len = np.mean(pred_len_list)
        early_stop_prob = early_stop_count / bs if bs > 0 else 0.0  # 提前终止概率

        # 打印新增提前终止概率
        print(f"Batch Metrics: P={batch_precision:.4f}, R={batch_recall:.4f}, F1={batch_f1:.4f}, AvgExtLen={avg_ext_len:.2f}, EarlyStopProb={early_stop_prob:.4f}")

        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com, step_rewards = [com[0]], []
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)
            repeat_count = 0

            for node in com[1:]:
                if node in temp_com and node != "Stp":
                    repeat_count += 1
                    repeat_penalty = -self.repeat_penalty_coeff * repeat_count
                    step_rewards.append(repeat_penalty)
                    continue

                if node == "Stp":
                    step_rewards.append(-0.1)
                    continue

                pre_f1 = eval_f1(temp_com, true_com_set)
                temp_com.append(node)
                curr_f1 = eval_f1(temp_com, true_com_set)

                if curr_f1 > pre_f1 and curr_f1 > self.min_f1_threshold:
                    base_reward = curr_f1 * self.f1_base_weight
                else:
                    base_reward = -(1 - curr_f1) * self.f1_base_weight

                curr_pred_len = len(temp_com)
                if curr_pred_len > true_com_len:
                    base_reward *= self.len_penalty_coeff ** (curr_pred_len - true_com_len)

                base_reward = np.clip(base_reward, -1.0, 1.0)
                step_rewards.append(base_reward)

            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            else:
                discounted = [0.0]
            rewards_list.append(discounted)

        max_len_pad = max(max(len(r) for r in rewards_list), max(len(lp) for lp in logps)) if (rewards_list and logps) else 0
        rewards_padded = np.zeros((bs, max_len_pad))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, : len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        logps_padded = []
        for lp_list in logps:
            padded = [
                lp_list[j] if j < len(lp_list) and lp_list[j] is not None 
                else torch.tensor(-1e9, device=self.device, requires_grad=True)
                for j in range(max_len_pad)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        mask = (torch.arange(max_len_pad, device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)).float()

        recall_low = 0.4
        recall_mid = 0.7
        recall_high = 0.9

        if batch_recall < recall_low:
            loss_weight = 3.0
            recall_bonus = 0.0
        elif batch_recall < recall_mid:
            loss_weight = 2.0
            recall_bonus = 0.1
        elif batch_recall < recall_high:
            loss_weight = 1.5
            recall_bonus = 0.2
        else:
            loss_weight = 1.0
            recall_bonus = 0.3

        base_loss = -(rewards.detach() * logps * mask).sum()
        final_loss = (base_loss * loss_weight) - (recall_bonus * bs)

        print(f"Loss Adjust | Base Loss: {base_loss.item():.4f}, Weight: {loss_weight:.1f}, Bonus: {recall_bonus:.1f}, Final Loss: {final_loss.item():.4f}")

        self.optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        param_mean = torch.mean(torch.stack([p.data.mean() for p in self.model.parameters() if p.requires_grad]))
        grad_mean = torch.mean(torch.stack([p.grad.mean() if p.grad is not None else torch.tensor(0.0, device=self.device) for p in self.model.parameters() if p.requires_grad]))
        print(f"\nGrad Check | Norm: {grad_norm:.4f} | Param Mean: {param_mean:.4f} | Grad Mean: {grad_mean:.4f}")
        print(f"Final Loss: {final_loss.item():.4f}")

        return final_loss.item()