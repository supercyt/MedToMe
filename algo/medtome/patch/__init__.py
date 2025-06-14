# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .deit import apply_patch as deit
from .aug import apply_patch as aug

__all__ = ["deit", "aug"]
