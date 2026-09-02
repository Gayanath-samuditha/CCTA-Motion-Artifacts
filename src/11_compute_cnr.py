
#s1_p92_cnr.py
#S1 = mean HU of voxels in TotalSeg coronary mask where HU>=200 and HU<=P92
#CNR = |S1 - S2| / sigma_IVS
#S2 = mean HU of myocardium excluding septum
#sigma = std HU of interventricular septum (septum_smart_v4)
#Resumable - skips cases already marked ok in existing output CSV
#Saves checkpoint every 50 cases

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

CT_DIR    = "/wrk-vakka/users/mohogaya/caisa/data/datasets/imagecas/niigz"
COR_DIR   = "/wrk-vakka/users/mohogaya/caisa/data/totalsegmentator_results1/coronary_arteries"
HEART_DIR = "/wrk-vakka/users/mohogaya/caisa/data/totalsegmentator_results/heartchambers_highres"
SEPT_DIR  = "/wrk-vakka/users/mohogaya/caisa/data/septum_smart_v4"
OUT_CSV   = "/wrk-vakka/users/mohogaya/caisa/results/s1_p92_cnr.csv"
HU_FLOOR  = 200.0
PERCENTILE = 92
SAVE_EVERY = 50


def compute_s1(hu_all, floor, pct):
    #apply floor
    gated = hu_all[hu_all >= floor]
    if len(gated) == 0:
        return None, None
    #apply upper bound at given percentile
    upper = float(np.percentile(gated, pct))
    trimmed = gated[gated <= upper]
    if len(trimmed) == 0:
        return None, None
    return round(float(np.mean(trimmed)), 4), round(upper, 4)


def compute_s2_sigma(ct, case_id):
    myo_path  = Path(HEART_DIR) / case_id / "heart_myocardium.nii.gz"
    sept_path = Path(SEPT_DIR)  / f"{case_id}_septum.nii.gz"
    if not myo_path.exists():
        return None, None, "missing_myo"
    if not sept_path.exists():
        return None, None, "missing_septum"
    myo  = nib.load(myo_path).get_fdata().astype(bool)
    sept = nib.load(sept_path).get_fdata().astype(bool)
    myo_hu  = ct[myo & ~sept]
    sept_hu = ct[sept]
    if len(myo_hu) == 0:
        return None, None, "empty_myo"
    if len(sept_hu) == 0:
        return None, None, "empty_septum"
    s2    = round(float(np.mean(myo_hu)), 4)
    sigma = round(float(np.std(sept_hu)), 4)
    return s2, sigma, "ok"


def process(case_id):
    try:
        ct_id    = case_id.replace("ImageCAS_", "")
        ct_path  = Path(CT_DIR) / f"{ct_id}.img.nii.gz"
        cor_path = Path(COR_DIR) / case_id / "coronary_arteries.nii.gz"
        if not ct_path.exists():
            return {"case_id": case_id, "status": "missing_ct"}
        if not cor_path.exists():
            return {"case_id": case_id, "status": "missing_coronary"}

        ct  = nib.load(ct_path).get_fdata().astype(np.float32)
        cor = nib.load(cor_path).get_fdata().astype(bool)
        if cor.sum() == 0:
            return {"case_id": case_id, "status": "empty_mask"}

        hu_all = ct[cor]
        s1, p92_val = compute_s1(hu_all, HU_FLOOR, PERCENTILE)
        if s1 is None:
            return {"case_id": case_id, "status": "empty_after_filter"}

        s2, sigma, status = compute_s2_sigma(ct, case_id)
        if status != "ok":
            return {"case_id": case_id, "status": status}

        cnr = round(abs(s1 - s2) / sigma, 4) if sigma > 0 else None

        print(f"Done: {case_id}  s1={s1}  p92={p92_val}  s2={s2}  sigma={sigma}  cnr={cnr}", flush=True)
        return {
            "case_id":   case_id,
            "s1_p92":    s1,
            "p92_upper": p92_val,
            "s2_myo":    s2,
            "sigma_ivs": sigma,
            "cnr":       cnr,
            "status":    "ok"
        }
    except Exception as e:
        print(f"Failed: {case_id} - {e}", flush=True)
        return {"case_id": case_id, "status": f"error: {str(e)[:80]}"}


def save(existing, new, path):
    pd.DataFrame(existing + new).to_csv(path, index=False, float_format="%.4f")


def main():
    cases = sorted([f"ImageCAS_{i:04d}" for i in range(1, 1001)])

    #resume support
    existing_rows, done_ids = [], set()
    out_path = Path(OUT_CSV)
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        done_ok     = existing_df[existing_df["status"] == "ok"]
        done_ids    = set(done_ok["case_id"])
        existing_rows = existing_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} already done", flush=True)

    todo = [c for c in cases if c not in done_ids]
    print(f"{len(cases)} total | {len(done_ids)} done | {len(todo)} to process", flush=True)

    new_results = []
    for i, case_id in enumerate(todo):
        new_results.append(process(case_id))
        if (i + 1) % SAVE_EVERY == 0:
            save(existing_rows, new_results, out_path)
            print(f"...checkpoint {i+1}/{len(todo)}", flush=True)

    save(existing_rows, new_results, out_path)
    df  = pd.read_csv(out_path)
    ok  = (df["status"] == "ok").sum()
    print(f"Done: {ok}/{len(df)} - {out_path}", flush=True)


if __name__ == "__main__":
    main()
