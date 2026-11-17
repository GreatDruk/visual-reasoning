"""Use val labels only to measure which scene hypotheses the renderers can express."""

from pathlib import Path
import sys

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd()))
from native_centers import FIELDS, GROUPS, read_rows
from src.models.slot_program import EXACT
from src.scene_posterior import _draws, _execute
from src.twins import find_twins


if __name__ == "__main__":
    directory = Path("artefacts/research")
    frame, scenes, captions, programs = read_rows("val")
    rows = frame.label.to_numpy().astype(bool)
    base = [
        np.load(directory/f"region_{name}.npz")["joint"]
        for name in ("primitives","seams","count_supervision")
    ]
    scores = np.mean(base,axis=0)
    for kind in ("silhouette","seams","counts"):
        with np.load(directory/f"scene_{kind}.npz") as source:
            table = {key:source[key] for key in ("valid","p","xy","offsets",*FIELDS)}
            assert np.array_equal(source["ids"],frame.id)
        no_joint = 0
        no_joint_on_errors = 0
        total_error_scenes = 0
        supported = 0
        count_cases = 0
        pairs_scores = []
        for i, ix in enumerate(scenes.values()):
            keep = table["valid"][i]
            p, xy = table["p"][i,keep], table["xy"][i,keep]
            if not len(p):
                continue
            u = _draws(len(p),8,42,4096)
            present = p > u[...,0]
            code = np.zeros_like(present,dtype=np.int32)
            for j,(group,attr) in enumerate(zip(GROUPS,FIELDS),1):
                distr = table[attr][i,keep]
                chosen = (u[...,j,None] > distr.cumsum(-1)).sum(-1).clip(max=len(group)-1)
                code |= 1<<group[chosen]
            local = [programs[n] for n in ix]
            y = rows[ix]
            hypothesis = _execute(local,code,present,xy[None]+logit(u[...,-3:-1])*.026,p,
                                  table["offsets"],u[...,-1])
            miss = not ((hypothesis==y).all(1)).any()
            wrong = ((2*y.astype(float)-1)*scores[ix] < 0).any()
            no_joint += miss
            no_joint_on_errors += miss and wrong
            total_error_scenes += wrong
            total = next((q.number for q,t in zip(local,y) if t and q.op==EXACT and
                          not q.a and not q.relation),None)
            if total is None:
                continue
            count_cases += 1
            matches = present.sum(1)==total
            supported += matches.any()
            subset = frame.iloc[ix]
            pairs = find_twins(subset.image,[captions[j] for j in ix],subset.category)
            linked = hypothesis[:,pairs[:,0]] ^ hypothesis[:,pairs[:,1]]
            weights = np.exp(1.3*(linked.sum(1)-len(pairs)))
            for j,(a,b) in enumerate(pairs):
                if any(q.op==EXACT and not q.a and not q.relation for q in (local[a],local[b])):
                    continue
                keep_pair = linked[:,j] * weights
                selected = keep_pair*matches
                if not selected.any():
                    selected = keep_pair
                baseline = ((hypothesis[:,a]*keep_pair).sum()+3)/(keep_pair.sum()+6)
                updated = ((hypothesis[:,a]*selected).sum()+3)/(selected.sum()+6)
                pairs_scores.extend([(y[a],baseline,updated),(y[b],1-baseline,1-updated)])
        truth, baseline, oracle = np.asarray(pairs_scores).T
        print(kind,"no joint",no_joint,"on error scenes",no_joint_on_errors,total_error_scenes,"count support",
              supported,count_cases,"paired non-count AUC",roc_auc_score(truth,baseline),
              roc_auc_score(truth,oracle),flush=True)
