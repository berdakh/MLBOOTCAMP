"""scaleml - Level 3's toolkit: deep learning and large models on one GPU.

Core (shared by every Level 3 notebook):

    device.py    device / precision selection, size budgets, timing, memory
    train.py     an explicit training loop: AMP, accumulation, clipping, checkpoints
    textdata.py  public Hugging Face text datasets, sampled to a budget

Topic modules are added by the notebooks that introduce them; exercise graders
live in the ``grading`` package (src/grading/chNN.py) and reuse
``churnlab.checks.run_check``.

Level 3 targets a free Colab T4 GPU. On a CPU every notebook still runs, with
the smaller sizes chosen by ``device.budget()``.
"""

__version__ = "3.0.0"
