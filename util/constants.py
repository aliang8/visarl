import os

ROOT = "/ROOT/PATH/"
REPLAY_BUFFER_DIR = os.path.join(ROOT, "replay_buffers")
SRC = os.path.join(ROOT, "src")
MMAE = os.path.join(ROOT, "MultiMAE")
encoder_to_dim_mapping = {"r3m_rn50": 2048, "r3m_rn34": 512, "mmae": 256}
