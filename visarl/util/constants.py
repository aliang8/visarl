import os

ROOT_DIR = "/data/anthony/visarl"
REPLAY_BUFFER_DIR = os.path.join(ROOT_DIR, "replay_buffers")
MMAE = "/data/anthony/MultiMAE"
encoder_to_dim_mapping = {"r3m_rn50": 2048, "r3m_rn34": 512, "mmae": 256}
