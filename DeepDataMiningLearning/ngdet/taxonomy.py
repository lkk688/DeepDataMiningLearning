"""
ngdet.taxonomy
==============

Configurable *unified label space* for cross-dataset 2D detection evaluation.

WHY THIS FILE EXISTS
--------------------
Every detector and every dataset speaks a different "class language":

    * COCO-pretrained DETR / YOLO  -> 80 COCO classes (person, car, truck, bus, ...)
    * KITTI ground truth           -> Car, Van, Truck, Pedestrian, Cyclist, Tram, ...
    * Waymo ground truth           -> Vehicle, Pedestrian, Cyclist, Sign
    * NuScenes ground truth        -> car, truck, bus, pedestrian, bicycle, barrier, ...

To compare a model's predictions against a dataset's ground truth we must first
project BOTH into a common, *configurable* taxonomy (e.g. a coarse 3-class driving
taxonomy: vehicle / person / cyclist). This file owns that projection.

KEY DESIGN CHOICE
-----------------
We map by **class NAME (string)**, not by integer id. Different models use
different id orderings (DETR uses the 91-entry COCO index with N/A gaps; Ultralytics
YOLO uses 80 contiguous ids), but they all expose human-readable names via
`model.config.id2label` / `model.names`. Mapping by lowercased name is therefore the
single robust bridge that works for every detector and every dataset.

A class name that is not listed in the active taxonomy is mapped to `None`
(i.e. "ignore / drop") -- e.g. COCO `traffic light`, NuScenes `barrier`,
Waymo `Sign` are dropped when evaluating a driving vehicle/person/cyclist taxonomy.

HOW TO ADD / SWITCH TAXONOMIES
------------------------------
Add an entry to `TAXONOMY_PRESETS` below, or pass your own dict to
`Taxonomy.from_synonyms(...)`. Each preset maps:

    unified_class_name -> [list of source synonyms, lowercased]

The unified class id is simply the index of the unified class name in the preset
(insertion order is preserved, so the first key gets id 0, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Taxonomy presets: unified_name -> synonyms that should fold into it.
# Synonyms are matched case-insensitively against detector/dataset class names.
# ---------------------------------------------------------------------------
TAXONOMY_PRESETS: Dict[str, Dict[str, List[str]]] = {
    # Coarse 3-class driving taxonomy (recommended default, most robust for
    # zero-shot cross-dataset comparison).
    "driving3": {
        "vehicle": [
            "vehicle", "car", "truck", "bus", "van", "trailer",
            "construction_vehicle", "construction vehicle", "tram", "train",
            "motorcycle", "motorbike",
        ],
        "person": [
            "person", "pedestrian", "person_sitting", "person sitting",
        ],
        # NOTE: this is a deliberate, documented simplification. KITTI/Waymo
        # "Cyclist" annotates the rider+bike as one box, whereas COCO annotates
        # "bicycle" (the object) and "person" (the rider) separately. We fold
        # the COCO "bicycle" object into "cyclist" so the taxonomies line up;
        # be aware this is not a perfect bijection and slightly favors models
        # that emit a single bike box. Discuss this caveat when reporting.
        "cyclist": [
            "cyclist", "bicycle", "bike",
        ],
    },

    # Two-class taxonomy: only the two categories every AD dataset agrees on.
    "vehicle_person": {
        "vehicle": [
            "vehicle", "car", "truck", "bus", "van", "trailer",
            "construction_vehicle", "tram", "train", "motorcycle", "motorbike",
        ],
        "person": ["person", "pedestrian", "person_sitting"],
    },

    # Finer 5-class taxonomy (use when you want to expose where coarse mapping
    # hides capability differences). "bus"/"truck" kept distinct from "car".
    "driving5": {
        "car": ["car", "vehicle", "van"],
        "truck": ["truck", "trailer", "construction_vehicle", "construction vehicle"],
        "bus": ["bus"],
        "person": ["person", "pedestrian", "person_sitting"],
        "cyclist": ["cyclist", "bicycle", "bike", "motorcycle", "motorbike"],
    },
}


# ---------------------------------------------------------------------------
# Curated prompt synonyms for OPEN-VOCABULARY detectors.
#
# Open-vocab models (Grounding DINO, OWLv2, LocateAnything) are far more accurate
# when prompted with concrete object words than with an abstract unified-class
# name. e.g. prompting "vehicle" gives poor recall, while prompting
# "car . truck . bus" works well. These terms are the prompt vocabulary; each
# term still maps back to a unified id (the value's index in the preset).
# ---------------------------------------------------------------------------
PROMPT_SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "driving3": {
        "vehicle": ["car", "truck", "bus", "van"],
        "person": ["person", "pedestrian"],
        "cyclist": ["bicycle", "cyclist"],
    },
    "vehicle_person": {
        "vehicle": ["car", "truck", "bus", "van"],
        "person": ["person", "pedestrian"],
    },
    "driving5": {
        "car": ["car"],
        "truck": ["truck"],
        "bus": ["bus"],
        "person": ["person", "pedestrian"],
        "cyclist": ["bicycle", "cyclist"],
    },
}


@dataclass
class Taxonomy:
    """A unified label space plus a name->id lookup with synonym folding.

    Attributes
    ----------
    name : str
        Preset name (for logging / output filenames).
    classes : List[str]
        Ordered unified class names; index == unified class id.
    _syn2id : Dict[str, int]
        Lowercased source-synonym -> unified id. Built from the preset.
    """

    name: str
    classes: List[str]
    _syn2id: Dict[str, int]
    #: unified_name -> curated open-vocab prompt terms (empty -> use class names)
    prompt_syn: Dict[str, List[str]] = None

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_preset(cls, preset: str = "driving3") -> "Taxonomy":
        if preset not in TAXONOMY_PRESETS:
            raise KeyError(
                f"Unknown taxonomy preset '{preset}'. "
                f"Available: {list(TAXONOMY_PRESETS)}"
            )
        return cls.from_synonyms(preset, TAXONOMY_PRESETS[preset],
                                 prompt_syn=PROMPT_SYNONYMS.get(preset))

    @classmethod
    def from_synonyms(cls, name: str, synonyms: Dict[str, List[str]],
                      prompt_syn: Dict[str, List[str]] = None) -> "Taxonomy":
        """Build from a {unified_name: [synonyms]} dict (insertion order = id)."""
        classes = list(synonyms.keys())
        syn2id: Dict[str, int] = {}
        for uid, uname in enumerate(classes):
            # the unified name itself is always a valid synonym
            syn2id[uname.lower()] = uid
            for s in synonyms[uname]:
                syn2id[s.lower()] = uid
        # prompt terms are also valid synonyms for mapping returned labels back
        if prompt_syn:
            for uname, terms in prompt_syn.items():
                uid = classes.index(uname)
                for t in terms:
                    syn2id.setdefault(t.lower(), uid)
        return cls(name=name, classes=classes, _syn2id=syn2id, prompt_syn=prompt_syn)

    # -- queries -------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def name_to_id(self, class_name: str) -> Optional[int]:
        """Map a single source class name to a unified id, or None if it should
        be ignored (not part of this taxonomy)."""
        if class_name is None:
            return None
        return self._syn2id.get(str(class_name).strip().lower(), None)

    def build_id_lut(self, source_id2name: Dict[int, str]) -> Dict[int, Optional[int]]:
        """Pre-compute a {source_class_id -> unified_id_or_None} lookup table.

        Detectors and datasets both expose an id->name dict; call this once at
        construction time so per-detection mapping is a cheap dict lookup.
        """
        return {sid: self.name_to_id(sname) for sid, sname in source_id2name.items()}

    def prompts(self) -> List[str]:
        """Bare class-name prompts (one per unified id). Simple but low-recall for
        abstract names like 'vehicle'; prefer `open_vocab_terms()` for real runs."""
        return list(self.classes)

    def open_vocab_terms(self):
        """Curated concrete prompt terms for open-vocab detectors.

        Returns a flat list of (term, unified_id) pairs, e.g.
        [("car",0),("truck",0),("bus",0),("van",0),("person",1),...]. Falls back
        to the bare class names if the taxonomy has no curated prompt synonyms.
        Use this to build a Grounding-DINO caption or an OWL query list while still
        knowing each term's unified class.
        """
        if not self.prompt_syn:
            return [(c, i) for i, c in enumerate(self.classes)]
        pairs = []
        for uid, uname in enumerate(self.classes):
            for term in self.prompt_syn.get(uname, [uname]):
                pairs.append((term, uid))
        return pairs

    def open_vocab_terms_by_class(self):
        """Curated prompt terms grouped per unified class: {unified_id: [terms]}.
        Used by detectors that prompt once per class (e.g. LocateAnything)."""
        if not self.prompt_syn:
            return {i: [c] for i, c in enumerate(self.classes)}
        return {uid: list(self.prompt_syn.get(uname, [uname]))
                for uid, uname in enumerate(self.classes)}


# ---------------------------------------------------------------------------
# Convenience: source taxonomies for the datasets we ship adapters for.
# These let datasets.py turn dataset-native label ids into names, which are then
# folded into the active unified taxonomy via Taxonomy.build_id_lut.
# (Kept here so all "class language" knowledge lives in one file.)
# ---------------------------------------------------------------------------

# KITTI label ids as used by detection/dataset_kitti.py (background=0).
KITTI_ID2NAME = {
    0: "__background__", 1: "Car", 2: "Van", 3: "Truck", 4: "Pedestrian",
    5: "Person_sitting", 6: "Cyclist", 7: "Tram", 8: "Misc", 9: "DontCare",
}

# Waymo type ids as used by detection/dataset_waymov3_1.py (WAYMO_CLASSES).
WAYMO_ID2NAME = {0: "Unknown", 1: "Vehicle", 2: "Pedestrian", 3: "Cyclist", 4: "Sign"}

# NuScenes simplified category ids as used by detection/dataset_nuscenes.py
# (CATEGORY_NAMES, index == label id).
NUSCENES_ID2NAME = {
    0: "car", 1: "truck", 2: "bus", 3: "trailer", 4: "construction_vehicle",
    5: "pedestrian", 6: "motorcycle", 7: "bicycle", 8: "traffic_cone", 9: "barrier",
}


# ===========================================================================
# HOW TO TEST / RUN THIS FILE
# ===========================================================================
# From the repo root (the dir that contains the DeepDataMiningLearning package):
#
#   python -m DeepDataMiningLearning.ngdet.taxonomy
#
# Expected: prints the driving3 taxonomy, shows how KITTI/COCO-ish names fold
# into unified ids, and demonstrates the open-vocab prompt list.
# ===========================================================================
if __name__ == "__main__":
    tax = Taxonomy.from_preset("driving3")
    print(f"Taxonomy '{tax.name}': {tax.num_classes} classes -> {tax.classes}")
    print("Open-vocab prompts:", tax.prompts())

    print("\nFolding KITTI ground-truth names into unified ids:")
    for sid, sname in KITTI_ID2NAME.items():
        print(f"  KITTI[{sid}] {sname:15s} -> {tax.name_to_id(sname)}")

    print("\nFolding a few COCO detector names into unified ids:")
    for sname in ["person", "car", "truck", "bus", "bicycle", "motorcycle",
                  "traffic light", "dog"]:
        print(f"  COCO {sname:15s} -> {tax.name_to_id(sname)}")

    print("\nSwitching taxonomy to 'driving5':")
    tax5 = Taxonomy.from_preset("driving5")
    print(f"  classes -> {tax5.classes}")
    for sname in ["car", "truck", "bus", "pedestrian", "bicycle"]:
        print(f"  {sname:12s} -> id {tax5.name_to_id(sname)} "
              f"({tax5.classes[tax5.name_to_id(sname)]})")
