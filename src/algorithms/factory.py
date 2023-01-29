from src.algorithms.sac import SAC
from src.algorithms.sac_masked_obs import SACMaskedObs
from src.algorithms.sac_fixed_rep import SAC_MAE

# from src.offline_rl.conservative_sac import ConservativeSAC

algorithm = {
    "sac": SAC,
    "sac_masked_obs": SACMaskedObs,
    "sac_fixed_rep": SAC_MAE,
    # "cql": ConservativeSAC,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
