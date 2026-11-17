"""Neural Network Lab: look inside Team 1's pretrained weld-anomaly autoencoder.

Everything shown by this module comes from real inference with the published weights
(Hugging Face: iaac-maai/hackathon-team1-weld-anomaly-ae). Nothing is retrained.
"""
import os
import sys

# Anaconda on Windows ships a second copy of the OpenMP runtime (libiomp5md.dll) next to the one inside
# PyTorch, and the process aborts with "OMP: Error #15" as soon as both load. The team notebook kernel
# already runs with this flag. It must be set BEFORE torch / numpy are imported.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
