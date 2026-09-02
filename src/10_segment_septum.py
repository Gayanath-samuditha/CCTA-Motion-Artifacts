#septum_smart.py
#Watershed-based IVS segmentation with anatomical postprocessing
#RV-bias fix applied: watershed landscape uses signed distance (dist_lv - dist_rv)
#so the boundary falls at the true equidistant midline between the ventricles.
#Usage: python septum_smart.py ct_dir seg_dir out_dir --workers 4

import argparse
import warnings
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from pathlib import Path
from multiprocessing import Pool
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def get_spacing(img):
    return np.array(img.header.get_zooms()[:3], dtype=float)


def gpu_erode(vol, device):
    t = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    k = torch.ones(1, 1, 3, 3, 3, device=device)
    out = F.conv3d(t, k, padding=1)
    return (out.squeeze().cpu().numpy() >= 27).astype(bool)


def make_sphere_kernel(radius_mm, spacing):
    rz = int(np.ceil(radius_mm / spacing[2]))
    ry = int(np.ceil(radius_mm / spacing[1]))
    rx = int(np.ceil(radius_mm / spacing[0]))
    z, y, x = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    return ((z * spacing[2])**2 +
            (y * spacing[1])**2 +
            (x * spacing[0])**2) <= radius_mm**2


def smart_postprocess(septum_raw, lv, rv, myo, spacing,
                      close_mm=3.0, bridge_mm=8.0,
                      min_blob_mm3=100.0,
                      valid_lv_dist_mm=15.0,
                      valid_rv_dist_mm=15.0):
    sept = septum_raw.copy()

    if close_mm > 0:
        k = make_sphere_kernel(close_mm, spacing)
        sept = ndi.binary_closing(sept, structure=k) & myo

    sept = ndi.binary_fill_holes(sept) & myo

    labeled, n_cc = ndi.label(sept)
    if n_cc == 0:
        return sept

    dist_lv = ndi.distance_transform_edt(~lv, sampling=spacing)
    dist_rv = ndi.distance_transform_edt(~rv, sampling=spacing)
    vox_vol = float(np.prod(spacing))
    min_vox = int(np.ceil(min_blob_mm3 / vox_vol))

    valid_labels = []
    for i in range(1, n_cc + 1):
        comp = (labeled == i)
        if int(comp.sum()) < min_vox:
            continue
        if (float(dist_lv[comp].mean()) <= valid_lv_dist_mm and
                float(dist_rv[comp].mean()) <= valid_rv_dist_mm):
            valid_labels.append(i)

    if len(valid_labels) == 0:
        sizes = [(int((labeled == i).sum()), i) for i in range(1, n_cc + 1)]
        valid_labels = [max(sizes)[1]]

    valid_mask = np.isin(labeled, valid_labels)

    if len(valid_labels) > 1 and bridge_mm > 0:
        bk = make_sphere_kernel(bridge_mm, spacing)
        valid_mask = ndi.binary_closing(valid_mask, structure=bk) & myo

    sept = (ndi.binary_fill_holes(valid_mask) & myo).astype(bool)

    #step6- clip septum to Z range where both ventricles co-exist
    lv_z = np.where(lv.any(axis=(1,2)))[0]
    rv_z = np.where(rv.any(axis=(1,2)))[0]
    if len(lv_z) > 0 and len(rv_z) > 0:
        z_lo = max(lv_z.min(), rv_z.min())
        z_hi = min(lv_z.max(), rv_z.max())
        sept[:z_lo]   = False
        sept[z_hi+1:] = False
    return sept


def process(args):
    case_id, ct_dir, seg_dir, out_dir, dev_str, hu_min, hu_max, sep_mm, \
        close_mm, bridge_mm = args

    out_path = Path(out_dir) / f'{case_id}_septum.nii.gz'
    if out_path.exists():
        return

    try:
        dev = torch.device(dev_str if torch.cuda.is_available() else 'cpu')
        sp  = Path(seg_dir) / case_id

        ref     = nib.load(sp / 'heart_ventricle_left.nii.gz')
        lv      = ref.get_fdata().astype(bool)
        rv      = nib.load(sp / 'heart_ventricle_right.nii.gz').get_fdata().astype(bool)
        myo     = nib.load(sp / 'heart_myocardium.nii.gz').get_fdata().astype(bool)
        spacing = get_spacing(ref)

        ct_id = case_id.replace('ImageCAS_', '')
        ct    = nib.load(Path(ct_dir) / f'{ct_id}.img.nii.gz').get_fdata()

        lv_dil  = ndi.binary_dilation(lv, iterations=3) & myo
        rv_dil  = ndi.binary_dilation(rv, iterations=3) & myo
        markers = np.zeros(myo.shape, dtype=np.int32)
        markers[lv_dil] = 1
        markers[rv_dil] = 2

        dist_lv   = ndi.distance_transform_edt(~lv, sampling=spacing)
        dist_rv   = ndi.distance_transform_edt(~rv, sampling=spacing)
        landscape = dist_lv - dist_rv
        ws        = watershed(landscape, markers, mask=myo)

        lv_reg = ws == 1
        rv_reg = ws == 2

        dil_px  = max(1, int(round(sep_mm / spacing.mean())))
        lv_bnd  = ndi.binary_dilation(lv_reg, iterations=dil_px) & myo
        rv_bnd  = ndi.binary_dilation(rv_reg, iterations=dil_px) & myo
        septum  = lv_bnd & rv_bnd & myo

        septum = septum & ~ndi.binary_erosion(lv, iterations=2)
        septum = septum & ~ndi.binary_erosion(rv, iterations=2)

        septum = septum & (ct >= hu_min) & (ct <= hu_max)

        septum = smart_postprocess(
            septum_raw=septum, lv=lv, rv=rv, myo=myo,
            spacing=spacing, close_mm=close_mm, bridge_mm=bridge_mm,
        )

        #step6 — clip to Z range where BOTH ventricles co-exist
        lv_z = np.where(lv.any(axis=(1,2)))[0]
        rv_z = np.where(rv.any(axis=(1,2)))[0]
        if len(lv_z) > 0 and len(rv_z) > 0:
            z_lo = int(max(lv_z.min(), rv_z.min()))
            z_hi = int(min(lv_z.max(), rv_z.max()))
            septum[:z_lo]    = False
            septum[z_hi+1:]  = False

        nib.save(nib.Nifti1Image(septum.astype(np.uint8),
                                  ref.affine, ref.header), out_path)
        print(f'Done: {case_id}  vox={int(septum.sum())}')

    except Exception as e:
        print(f'Failed: {case_id} — {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ct_dir')
    p.add_argument('seg_dir')
    p.add_argument('out_dir')
    p.add_argument('--workers',   type=int,   default=4)
    p.add_argument('--device',    default='cpu')
    p.add_argument('--hu_min',    type=float, default=30.0)
    p.add_argument('--hu_max',    type=float, default=170.0)
    p.add_argument('--sep_mm',    type=float, default=9.0)
    p.add_argument('--close_mm',  type=float, default=3.0)
    p.add_argument('--bridge_mm', type=float, default=8.0)
    p.add_argument('--start',     type=int,   default=None)
    p.add_argument('--end',       type=int,   default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_cases = sorted([d.name for d in Path(args.seg_dir).iterdir() if d.is_dir()])
    if args.start and args.end:
        cases = [c for c in all_cases if args.start <= int(c.replace('ImageCAS_','')) <= args.end]
    else:
        cases = all_cases

    print(f'{len(cases)} cases')
    print(f'sep={args.sep_mm}mm  close={args.close_mm}mm  '
          f'bridge={args.bridge_mm}mm  hu={args.hu_min}-{args.hu_max}')

    task_args = [(c, args.ct_dir, args.seg_dir, args.out_dir, args.device,
                  args.hu_min, args.hu_max, args.sep_mm,
                  args.close_mm, args.bridge_mm) for c in cases]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            pool.map(process, task_args)
    else:
        for a in task_args:
            process(a)


if __name__ == '__main__':
    main()
