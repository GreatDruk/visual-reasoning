"""Refit the image and crop branches feeding the count_swap anchor."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from scripts.research import kaggle_suite, occlusion_verifier as verifier
from scripts.research import phase_refit, postrun_visual_audit
from kaggle_suite_common import CACHE, DATA, INPUT, REPORT, SCORES, WEIGHTS, load, prepare, setup
from kaggle_suite_crops import crop_extract, encode_tuned, train_pool
from kaggle_suite_detectors import detector_eval, train_detector


def setup_detectors():
    for directory in (CACHE,REPORT,SCORES,WEIGHTS):
        directory.mkdir(parents=True,exist_ok=True)
    for name in ("scene_silhouette","r50_img","r50_img_head","thaw","thaw_head","union_trainval"):
        shutil.copyfile(verifier.PRIORS/f"{name}.pt",WEIGHTS/f"{name}.pt")
    shutil.copyfile(INPUT/"scene_silhouette.pt",WEIGHTS/"scene_silhouette.pt")


def predict_union_part(shard,device):
    verifier.PRIORS=WEIGHTS
    verifier.CACHE.mkdir(exist_ok=True)
    phase_refit.extract("test",shard,device,single=True)


def combine(device):
    frame=pd.read_csv(DATA/"test.csv")
    def read(split,name,rows):
        with np.load(SCORES/f"{name}_{split}.npz") as saved:
            if not np.array_equal(saved["ids"],rows.id):
                raise ValueError(f"{name} scores have a different row order")
            return saved["score"]
    delta=postrun_visual_audit.rescue("test",128,device,weights=lambda n:load(WEIGHTS/f"{n}.pt"),
                                     scores_from=read,output_adjustment=True,data_path=DATA)
    # Val, train-only heads (r50 with TTA_WEIGHT .25): .5/.5/.25 gives AUC 0.999733 and 14 wrong
    # pairs; .65/.35/.35 gives 0.999783 and 11. The whole .6-.7 x .35-.5 region is >= 0.99976.
    occ=.65*read("test","r50_img",frame)+.35*read("test","sigft",frame)+.35*delta
    phase_refit.assemble("test","00")
    table=verifier.CACHE/"tables"
    shutil.copyfile(table/"test_00.npz",table/"test.npz")
    verifier.OUT=Path("union")
    verifier.OUT.mkdir(exist_ok=True)
    for seed in (421,853):
        source=INPUT/f"union_{seed}.pt"
        shutil.copyfile(source,verifier.OUT/f"control_{seed}.pt")
        verifier.infer("test","control",seed,device)
    values=[]
    for seed in (421,853):
        with np.load(verifier.OUT/f"control_{seed}_test.npz") as saved:
            np.testing.assert_array_equal(saved["ids"],frame.id)
            values.append(saved["score"])
    union=np.mean(values,axis=0)
    easy=(~frame.category.isin(["counting","spatial_counting","comparison"])) & (occ*union>0)
    result=(.75*occ+.25*union)*np.where(easy,1.5,1.)
    np.savez_compressed("union_tv_test.npz",ids=frame.id.to_numpy(),score=result)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("setup","prepare","fit","detect","crops","pool","encode",
                                      "fit_crop","score_crop","union","combine"))
    cli.add_argument("--split",choices=("train","test"),default="train")
    cli.add_argument("--arm",choices=("r50_img","thaw"),default="thaw")
    cli.add_argument("--shard",type=int,choices=(0,1),default=0)
    args=cli.parse_args()
    _,device=setup(421)
    if args.stage=="setup":
        setup_detectors()
    elif args.stage=="prepare":
        prepare(args.split,"sil")
    elif args.stage=="fit":
        train_detector(args.arm,warm_start=True,epochs_override=3,rate_scale=.25,batch_override=128)
    elif args.stage=="detect":
        detector_eval(args.split,args.arm)
    elif args.stage=="crops":
        crop_extract(args.split,args.shard,tuned_only=True)
    elif args.stage=="pool":
        train_pool()
    elif args.stage=="encode":
        encode_tuned(args.split,args.shard)
    elif args.stage=="fit_crop":
        kaggle_suite.fit_head("sigft")
    elif args.stage=="score_crop":
        kaggle_suite.head_eval("test","sigft")
    elif args.stage=="union":
        predict_union_part(args.shard,device)
    else:
        combine(device)


if __name__=="__main__":
    main()
