"""下载并验证检索模型：BGE-M3 (粗排) 与 bge-reranker-v2-m3 (精排)。

说明：
- 模型缓存到项目本地 ./models 目录，便于 demo 整体迁移。
- 通过 HuggingFace 官方源下载。如国内直连慢/失败，可设置环境变量加速：
    export HF_ENDPOINT=https://hf-mirror.com
  然后重新运行本脚本。
- 仅做下载与最小化推理验证，使用 CPU 即可，避免下载阶段占用 MPS。
"""
import os
import sys

# 模型缓存目录：项目内 ./models（优先级高于默认 ~/.cache/huggingface）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.environ.setdefault("HF_HOME", MODEL_DIR)
# 让 transformers / huggingface_hub 也使用该缓存
os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_DIR)
os.environ.setdefault("HF_HUB_CACHE", MODEL_DIR)

from FlagEmbedding import BGEM3FlagModel, FlagReranker


def download_bge_m3() -> BGEM3FlagModel:
    print("=" * 60)
    print("[1/2] 下载 BGE-M3 (粗排 encoder) ...")
    print("=" * 60)
    # use_fp16=False：CPU/Mac 下避免 fp16 问题；device 默认 cpu（无 CUDA）
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    # 触发一次编码，验证模型可用
    out = model.encode(["钢材采购合同履约验证"], return_dense=True)
    assert out is not None and "dense_vecs" in out
    print("BGE-M3 下载并验证完成。")
    return model


def download_reranker() -> FlagReranker:
    print("=" * 60)
    print("[2/2] 下载 bge-reranker-v2-m3 (精排 reranker) ...")
    print("=" * 60)
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    # 触发一次打分，验证模型可用
    score = reranker.compute_score([["逾期供货违约金", "若供方逾期交货，按日支付违约金。"]])
    print(f"  验证打分: {score}")
    print("bge-reranker-v2-m3 下载并验证完成。")
    return reranker


def main():
    print(f"模型缓存目录: {MODEL_DIR}")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    download_bge_m3()
    download_reranker()

    print("\n" + "=" * 60)
    print("全部模型已下载并验证通过。")
    print(f"缓存位置: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
