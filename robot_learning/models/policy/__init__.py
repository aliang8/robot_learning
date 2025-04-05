# from .action_chunking_transformer_decoder import ActionChunkingTransformerPolicy
# TODO: move this 
from hand_demos.models.act_policy import ACTPolicy

from .base import BasePolicy
from .mlp import MLPPolicy

POLICY_CLS_MAP = {
    "mlp": MLPPolicy,
    # "ac_decoder": ActionChunkingTransformerPolicy,
    "act_transformer": ACTPolicy,
}
