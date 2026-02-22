import sys
sys.path.insert(0, r"D:\CodeSummary\sci")  # 把项目根目录插入sys.path最顶端
from codes.components.nn.starter import run
import os
print("当前工作目录：", os.getcwd())  # 关键！决定了Python默认的搜索起点
def main():
    datasets=["PlusTokenPonzi","elliptic_txs","archive"]
    for d in datasets:
        run(dfname=d,seedNum=50)
main()