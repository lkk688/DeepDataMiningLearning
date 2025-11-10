# projects/freeze_utils.py
from typing import Iterable, Tuple
import re
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.dist import get_dist_info

@HOOKS.register_module()
class FreezeExceptHook(Hook):
    """Freeze all parameters except those whose names match allowlist patterns.

    - Sets `requires_grad=False` for all params not matching allowlist.
    - Optionally sets normalization layers to eval (freeze BN stats).
    - Prints a short summary once on rank0.

    Args:
        allowlist (Tuple[str, ...]): substrings or regex patterns; any match keeps trainable.
            e.g., ('view_transform',) will keep CrossAttnLSSTransform trainable only.
        freeze_norm (bool): if True, set norm layers (BN/LN/GN/IN) to eval & no grad.
        verbose (bool): log a brief summary on rank0.
        use_regex (bool): treat allowlist items as regex if True, else substring match.
    """
    def __init__(self,
                 allowlist: Tuple[str, ...] = ('view_transform',),
                 freeze_norm: bool = True,
                 verbose: bool = True,
                 use_regex: bool = False):
        self.allowlist = allowlist
        self.freeze_norm = freeze_norm
        self.verbose = verbose
        self.use_regex = use_regex
        self._done = False

    def _match(self, name: str) -> bool:
        if self.use_regex:
            return any(re.search(p, name) for p in self.allowlist)
        else:
            return any(p in name for p in self.allowlist)

    def before_train(self, runner) -> None:
        if self._done:
            return
        model = runner.model
        rank, _ = get_dist_info()

        # 1) Freeze / keep params
        total, keep, frozen = 0, 0, 0
        for n, p in model.named_parameters():
            total += 1
            if self._match(n):
                p.requires_grad = True
                keep += 1
            else:
                p.requires_grad = False
                frozen += 1

        # 2) Freeze norm layers' running stats (optional)
        if self.freeze_norm:
            norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm,
                          nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)
            for n, m in model.named_modules():
                if isinstance(m, norm_types) and not self._match(n):
                    m.eval()
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False

        if self.verbose and rank == 0:
            runner.logger.info(
                f"[FreezeExceptHook] total={total}, trainable={keep}, frozen={frozen}; "
                f"allowlist={self.allowlist}"
            )
        self._done = True