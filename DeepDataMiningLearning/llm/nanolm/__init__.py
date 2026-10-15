"""nanolm -- a config-driven language-model teaching lab (RNN -> char Transformer -> GPT -> parameter-golf).

Entry points:
    python -m nanolm.train   --config nanolm/configs/<exp>.yaml [key.sub=value ...]
    python -m nanolm.sweep   --sweep  nanolm/configs/sweeps/<sweep>.yaml
    python -m nanolm.analyze outputs/nanolm/<dir>
    python -m nanolm.prepare --config ... | --golf-shards N
See ../TUTORIAL_nanolm.md.
"""
