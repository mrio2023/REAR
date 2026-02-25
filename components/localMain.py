import sys
import os

# ========== 1. 固定项目根目录（sci），不受执行目录影响 ==========
# 方案：基于脚本位置动态计算，而非依赖sys.path
# 获取当前脚本（localMain.py）的绝对路径
current_file = os.path.abspath(__file__)
# 计算项目根目录（sci）：从当前脚本（components/localMain.py）向上回溯2级
# components → codes → sci
PROJECT_ROOT = os.path.abspath(os.path.join(current_file, '../../..'))
# 把项目根目录插入sys.path（保证模块导入正常）
sys.path.insert(0, PROJECT_ROOT)

# ========== 2. 强制切换工作目录到项目根目录（关键！统一路径基准） ==========
os.chdir(PROJECT_ROOT)
print("✅ 强制切换到项目根目录：", os.getcwd())  # 此时应该输出 D:\CodeSummary\sci

# ========== 3. 导入模块（此时路径已正确） ==========
from codes.components.nn.starter import run

# ========== 4. 定义主函数（保留你的业务逻辑） ==========
def main():
    datasets = ["elliptic2"]
    for d in datasets:
        run(dfname=d, seedNum=2, epochs=10, test_seed_num=2)

# ========== 5. 安全执行主函数（避免脚本被导入时自动执行） ==========
if __name__ == "__main__":
    main()