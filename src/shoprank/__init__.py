"""shoprank - Level 2's running project: ranking and recommendations at scale.

Core (shared by every Level 2 notebook):

    data.py      the ShopStream world, its logging policy and click simulation
    metrics.py   NDCG, MRR, recall@k, coverage
    store.py     where Level 2 artifacts live, and cached dataset loading

Topic modules (each introduced by its notebook) live alongside, and graders
for exercises live in the ``grading`` package (src/grading/chNN.py) and reuse
``churnlab.checks.run_check`` from Part I.
"""

__version__ = "2.0.0"
