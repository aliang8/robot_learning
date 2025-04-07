from hand_demos.models.act_policy import ACTPolicy
from hand_demos.models.action_chunking_transformer_decoder import (
    ActionChunkingTransformerPolicy,
)

from .mlp import MLPPolicy

POLICY_CLS_MAP = {
    "mlp": MLPPolicy,
    "ac_decoder": ActionChunkingTransformerPolicy,
    "act_transformer": ACTPolicy,
}
