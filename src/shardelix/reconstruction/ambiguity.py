"""Observable ambiguity report. Never equates apparent completeness with DPA."""
from collections import defaultdict
import hashlib
import numpy as np

def describe_ambiguity(pieces,diagnostics,rotations):
    classes=[]
    duplicate_count=0;groups_total=0
    if diagnostics.get("identical_piece_count",0):
        groups=defaultdict(list)
        for p in pieces:
            keys=[]
            for k in range(4 if rotations else 1):
                a=p.rotated(k);keys.append(hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).digest())
            groups[min(keys)].append(p.id)
        for ids in groups.values():
            if len(ids)<2:continue
            groups_total+=1;duplicate_count+=len(ids)
            if len(classes)<200:classes.append({"count":len(ids),"ids":ids[:200],"ids_truncated":len(ids)>200})
    slack=diagnostics.get("anchor_translation_slack",[0,0])
    return {"exact_equivalent_piece_count":duplicate_count,"exact_equivalence_classes":classes,
            "class_count":groups_total,"classes_truncated":groups_total>len(classes),
            "relative_translation_slack_yx":slack,"low_texture_piece_count":diagnostics.get("low_texture_piece_count"),
            "unanchored_component_count":diagnostics.get("unanchored_component_count",0),
            "accuracy_verified":False,
            "note":"Exact equivalence means identical observable normalized RGBA up to permitted rotations. Near-equivalence, hidden missing positions and absolute pose may still be ambiguous. This is not ground-truth accuracy."}
