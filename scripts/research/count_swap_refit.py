"""Continue fitting the actual count_swap branches on train + val."""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch

from scripts.research import (
    compare_spatial, count_evidence, curriculum_replay, curriculum_train,
    dense_counts, hidden_count_prior, rim_scene_unet, spatial_geometry, spatial_replay,
)
from src.twins import find_twins


RIM = curriculum_replay.crops
EXTRA = curriculum_train.data.EXTRA
SEEDS = (421, 853)
CONTOUR = "unet_boundary/views/new/0.55"


def train_models(root, seed, device):
    curriculum_train.OUT = root/"curriculum"
    curriculum_train.OUT.mkdir(parents=True, exist_ok=True)
    rim_scene_unet.OUT = curriculum_train.OUT
    initial = EXTRA/"initial"
    bank = curriculum_train.data.Data(device, all_scenes=True)
    timings = {}
    for name, batch, steps in (("unet_noexact",512,400),("r50_noexact",192,320),("unet_boundary",512,400)):
        arm, mode = name.split("_",1)
        start = time.monotonic()
        curriculum_train.fit(bank,arm,mode,seed,batch,steps,fixed=True,log_interval=200,
                              initial=initial/f"{name}_{seed}.pt",rate_scale=.25)
        timings[name]=time.monotonic()-start
    start = time.monotonic()
    rim_scene_unet.fit(seed,device,8,None,bank=bank.bank,batch=128,
                       initial=initial/f"scene_unet_{seed}.pt",rate_scale=.25)
    timings["scene_unet"]=time.monotonic()-start
    start = time.monotonic()
    RIM.v.OUT=root/"counts"/str(seed)
    RIM.v.OUT.mkdir(parents=True,exist_ok=True)
    dense_counts.fit(seed,device,6,scenes=bank.bank,initial=initial/f"counter_{seed}.pt",
                     batch=96,rate_scale=.25)
    timings["dense_r50"]=time.monotonic()-start
    curriculum_train.write_json(root/"curriculum"/f"times_{seed}.json",timings)


def train_geometry(root):
    frame=pd.read_csv(RIM.v.DATA/"train.csv")
    with np.load(EXTRA/"geometry_trainval.npz") as cache:
        fields=dict(cache)
    if not np.array_equal(fields["names"],frame.image.unique()):
        raise ValueError("Geometry training scenes differ from the combined split")
    spatial_geometry.fit(root,frame,fields)


def score_counter(root,seed,device):
    RIM.v.OUT=root/"counts"/str(seed)
    dense_counts.infer("test",seed,device)


def anchor(root,frame):
    path=root/"union_tv_test.npz"
    union=count_evidence.read_npz(path if path.is_file() else EXTRA/"union_tv_test.npz",frame)["score"]
    dense=[count_evidence.read_npz(root/"counts"/str(s)/f"control_{s}_test.npz",frame)["score"] for s in SEEDS]
    with np.load(EXTRA/"geometry_trainval.npz") as cache:
        background=cache["background"]
    prior=hidden_count_prior.placement_prior(EXTRA/"fragment_train.npz",background)
    hidden=hidden_count_prior.adjust(frame,union,prior,RIM.v.DATA/"test")
    active=frame.category.isin(dense_counts.ACTIVE).to_numpy()
    score=np.where(active,.75*union+.25*np.mean(dense,axis=0),union)+.5*(hidden-union)
    np.savez_compressed(root/"anchor_test.npz",ids=frame.id.to_numpy(),probability=expit(score))


def score_objects(root,frame,shard,seed,device):
    curriculum_replay.prepare(frame,root,shard,device)
    curriculum_replay.predict(frame,root,shard,device)
    probabilities=count_evidence.read_npz(root/"anchor_test.npz",frame)["probability"]
    color=curriculum_replay.scopes(frame)[0]&(np.abs(logit(probabilities))<=3)
    out,_=curriculum_replay.locations(root)
    if color.any():
        scene=frame[frame.image.isin(frame.image[color])].reset_index(drop=True)
        rim_scene_unet.infer(seed,device,"test",None,output=out,frame=scene,
                             checkpoint=root/"curriculum"/f"scene_unet_{seed}.pt")
    else:
        np.savez_compressed(out/f"scene_unet_{seed}_test.npz",ids=frame.id.to_numpy()[:0],score=np.empty(0))


def score_spatial(root,shard,device):
    spatial_replay.proposals(root,"test",shard,device)
    for seed in SEEDS:
        spatial_replay.infer(root,"test","unet_boundary",seed,shard,device)


def finish(root,frame,output,without_swap=None):
    spatial_geometry.infer(root,"test")
    f,base,_,active=spatial_replay.frames(root,"test")
    tables=spatial_replay.tables_for(root,"test")
    proposals=spatial_replay.candidates(root,"test","unet_boundary","views")
    contour,_=spatial_replay.replay(f,base,active,proposals,tables,.55,"new")
    np.savez_compressed(root/"spatial/test_branches.npz",ids=f.id.to_numpy(),keys=[CONTOUR],delta=contour[None])
    _,base,options=compare_spatial.build(root,"test")
    geo,full=(options[name] for name in (count_evidence.GEO_KEY,count_evidence.SPATIAL_KEY))
    mat=count_evidence.read_csv_scores(root/"shape_mat.csv",frame)
    probability=np.where(full==base,mat,expit(full))
    anchor_probability=count_evidence.read_npz(root/"anchor_test.npz",frame)["probability"]
    geom=count_evidence.read_npz(root/"spatial/test_geometry.npz",frame)
    dense=[count_evidence.read_npz(root/"counts"/str(s)/f"control_{s}_test.npz",frame)["score"] for s in SEEDS]
    twins=find_twins(frame.image,frame.caption,frame.category)
    scores,_=count_evidence.compose(frame,twins,(logit(anchor_probability),logit(mat),geo,full,probability,geom,dense))
    final=count_evidence.swap_counts(probability,scores["count"],twins)
    if not np.array_equal(np.sort(final),np.sort(probability)):
        raise ValueError("Swap changed the probability distribution")
    from scripts.research.compact_refit import write_scores
    write_scores(frame,final,output)
    if without_swap is not None:
        write_scores(frame,probability,without_swap)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("fit","geometry","count","anchor","objects","typed","spatial","export"))
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--seed",type=int,choices=SEEDS,default=421)
    cli.add_argument("--shard",type=int,choices=(0,1),default=0)
    cli.add_argument("--output",type=Path)
    cli.add_argument("--without-swap",type=Path)
    args=cli.parse_args()
    _,device=RIM.v.setup(args.seed)
    if args.stage=="fit":
        train_models(args.root,args.seed,device)
        return
    if args.stage=="geometry":
        train_geometry(args.root)
        return
    frame=pd.read_csv(RIM.v.DATA/"test.csv")
    if args.stage=="count":
        score_counter(args.root,args.seed,device)
    elif args.stage=="anchor":
        anchor(args.root,frame)
    elif args.stage=="objects":
        score_objects(args.root,frame,args.shard,args.seed,device)
    elif args.stage=="typed":
        curriculum_replay.export(frame,args.root)
    elif args.stage=="spatial":
        score_spatial(args.root,args.shard,device)
    else:
        if args.output is None:
            cli.error("export requires --output")
        finish(args.root,frame,args.output,args.without_swap)


if __name__=="__main__":
    main()
