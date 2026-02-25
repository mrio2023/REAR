
import sys
import os

# ========== 核心修改：适配Linux服务器路径 ==========
# 获取当前脚本（main.py）的绝对路径
current_file_path = os.path.abspath(__file__)
# 向上回溯找到sci目录（根据你的实际目录结构调整层级）
# 假设main.py路径：~/sci/codes/components/main.py
# 回溯层级：components → codes → sci（共3级）
sci_dir = os.path.abspath(os.path.join(current_file_path, '../../..'))
# 把sci目录插入sys.path最顶端（优先级最高）
sys.path.insert(0, sci_dir)

# 导入模块（现在能正确找到codes了）
from codes.components.nn.starter import run

# 打印关键路径信息，方便排查问题
print("项目根目录（sci）：", sci_dir)
print("当前工作目录：", os.getcwd())

def main():
    datasets=["PlusTokenPonzi","elliptic_txs","elliptic2"]
    for d in datasets:
        run(dfname=d,seedNum=50,epochs=40,test_seed_num=20)

if __name__ == "__main__":
    main()
