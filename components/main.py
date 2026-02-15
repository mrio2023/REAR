import pandas as pd
import random
from dataProcess import DataProcess
from nn.graph import Graph
import argparse
import datetime
from nn.detector import Detector
from nn.utils import seed_all, writerResToFile

def get_multimodal_seeds(g: Graph, search_size: int = 1):
    all_communities = []
    for tag in g.allNameTags:
        com_nodes = g.df_hacker[g.df_hacker["name_tag"] == tag]["address"].tolist()
        if com_nodes and len(com_nodes) > 1:
            all_communities.append(com_nodes)
    
    seeds = []
    com_indexs = []
    sample_num = min(search_size, len(all_communities))
    if sample_num == 0:
        print("警告：未找到有效社区，返回空种子节点")
        return [], []
    
    selected_coms = random.sample(all_communities, sample_num)
    for idx, com in enumerate(selected_coms):
        seed = random.choice(com)
        seeds.append(seed)
        com_indexs.append(idx)
    
    return seeds, com_indexs

def main(args):
    d = DataProcess(dfname=args.dataset)
    
    train_nodes, test_nodes, train_feature, test_feature, train_hacker, test_hacker = \
        d.train_nodes, d.test_nodes, d.train_feature, d.test_feature, d.train_hacker, d.test_hacker 
    
    g = Graph(dffeature=train_feature, dfhacker=train_hacker, dfnode=train_nodes)
    
    seeds, com_indexs = get_multimodal_seeds(g, args.search_size)
    if not seeds:
        print(f"数据集[{args.dataset}]无有效种子节点，跳过处理")
        return
    
    for i in range(len(seeds)):
        seed = seeds[i]
        com_index = com_indexs[i]
        print(f"\n===== 处理[{args.dataset}] - 第{i+1}个种子节点 =====")
        print(f"种子节点：{seed}，社区索引：{com_index}")
        
        try:
            detector = Detector(args, g)
            detector.oldSeed = seed
            
            if seed not in detector.old_to_new_node_mapping:
                print(f"警告: 种子节点 {seed} 不在映射中!")
                continue
            
            detector.seed = detector.old_to_new_node_mapping[seed]
            detector.com_index = com_index
            
            res = detector.detect()
            writerResToFile(args, res)
            print(f"节点[{seed}]处理完成，检测到社区节点数：{len(res[2])}")
            
        except Exception as e:
            print(f"节点[{seed}]处理失败！错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AscendEXHacker',
                        choices=['AscendEXHacker', 'elliptic_txs', 'PlusTokenPonzi', 'archive'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_ego_subG', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--g_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--search_size', type=int, default=1)
    parser.add_argument('--si', type=int, default=0.9)
    parser.add_argument('--resfileName', type=str, default='sp_cluster',
                        choices=['sp_cluster', 'KMedoids', 'Gmm', 'CengCi'])
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--k', type=int, default=1)

    args = parser.parse_args()
    seed_all(args.seed)

    print('= ' * 20)
    now = datetime.datetime.now()
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    datasets = ['AscendEXHacker', 'PlusTokenPonzi', 'archive']
    for dataset in datasets:
        args.dataset = dataset
        print(f"\n{'='*30} 开始处理数据集：{dataset} {'='*30}")
        main(args)

    print('\n' + '= ' * 20)
    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)