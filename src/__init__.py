"""src 包入口。

在导入任何子模块前设置线程数，规避 macOS 上 FAISS(OpenMP) 与
torch/MKL(BGE-M3、reranker) 多线程共存时的 segfault（exit 139）。
必须在 torch / faiss 被 import 之前执行，故放在包入口。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
