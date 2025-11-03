#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CT → sPET (NIfTI) **3D** pix2pix training + inference with visible progress.

Now using:
- 3D Generator: ResNet++ (SE residual blocks, 3D)
- 3D Discriminator: PatchGAN (shallower)

Data structure (paired):
train/
  CT/  *.nii or *.nii.gz
  PT/  *.nii or *.nii.gz     # same basenames as CT

For inference:
inference/
  CT/  *.nii or *.nii.gz     # generates sPET into --output_dir

Examples:
# Train (3D patches 64×64×64)
python -u spet_from_ct_3d.py --mode train --train_dir ./train --epochs 150 --batch_size 2 \
  --patch_size 64 64 64 --num_patches_per_epoch 4000 --num_workers 4

# Inference (center-crop stitching, context margin 12, overlap 0.75)
python -u spet_from_ct_3d.py --mode test --inference_dir ./inference --output_dir ./outputs \
  --checkpoints_dir ./checkpoints --checkpoint ./checkpoints/epoch_150.pt --patch_size 64 64 64 \
  --overlap 0.75 --context_margin 12
"""

import os
import glob
import argparse
import time
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------
# Utils: clipping & scaling
# ------------------------

CT_MIN, CT_MAX = -1000.0, 1000.0     # clip range for CT
PT_MIN, PT_MAX = 0.0, 30.0           # clip range for PET

def clip_scale_ct(ct):
    """Clip CT to [-1000,1000] and scale to [0,1]."""
    ct = np.clip(ct, CT_MIN, CT_MAX)
    return (ct - CT_MIN) / (CT_MAX - CT_MIN)

def descale_pet_from01(x01):
    """Back to PET scale [0,30] from [0,1]."""
    return x01 * (PT_MAX - PT_MIN) + PT_MIN

def clip_scale_pet(pt):
    """Clip PET to [0,30] and scale to [0,1]."""
    pt = np.clip(pt, PT_MIN, PT_MAX)
    return (pt - PT_MIN) / (PT_MAX - PT_MIN)

# ------------------------
# Data helpers
# ------------------------

def list_pairs(root_dir):
    ct_dir = os.path.join(root_dir, "CT")
    pt_dir = os.path.join(root_dir, "PT")
    assert os.path.isdir(ct_dir), f"Missing CT dir: {ct_dir}"
    assert os.path.isdir(pt_dir), f"Missing PT dir: {pt_dir}"
    exts = ("*.nii", "*.nii.gz")
    ct_paths = sorted([p for ext in exts for p in glob.glob(os.path.join(ct_dir, ext))])
    pairs = []
    for cpath in ct_paths:
        base = os.path.basename(cpath)
        ppath = os.path.join(pt_dir, base)
        if os.path.exists(ppath):
            pairs.append((cpath, ppath))
        else:
            raise FileNotFoundError(f"Missing PET for {base} at {ppath}")
    if len(pairs) == 0:
        raise RuntimeError("No pairs found in train/CT and train/PT")
    return pairs

def has_foreground(block_ct, block_pt, thr_ct=0.0, thr_pt=0.0):
    """Require both CT and PET have some > thr voxels to avoid empty background patches."""
    return (block_ct > thr_ct).any() and (block_pt > thr_pt).any()

def compute_valid_z_bands(ct: np.ndarray, pt: np.ndarray):
    """
    Compute axial slice validity and return list of (z_start, z_end_exclusive) bands
    where slices are valid. A slice is valid if:
      - PET slice is not all zeros, AND
      - CT slice is not entirely < 0.
    """
    D = ct.shape[0]
    valid = np.ones(D, dtype=bool)
    valid &= np.array([not np.all(pt[z] == 0) for z in range(D)], dtype=bool)
    valid &= np.array([not np.all(ct[z] < 0) for z in range(D)], dtype=bool)

    bands = []
    z = 0
    while z < D:
        if valid[z]:
            z0 = z
            while z < D and valid[z]:
                z += 1
            bands.append((z0, z))  # [z0, z)
        else:
            z += 1
    return bands, valid

# ------------------------
# 3D Patch Dataset (with pre-filtered slices)
# ------------------------

class PairedPatches3D(Dataset):
    """
    Samples 3D patches on-the-fly from paired CT/PT volumes.
    Before sampling, we remove axial slices where PET is all zeros or CT is entirely < 0.
    Patches are then sampled from the remaining valid Z-bands.
    """
    def __init__(self, train_dir, patch_size=(64,64,64), num_patches_per_epoch=4000, fg_ratio=0.7, attempts=30):
        super().__init__()
        self.pairs = list_pairs(train_dir)
        self.patch_size = tuple(patch_size)  # (D,H,W)
        self.num_samples = int(num_patches_per_epoch)
        self.fg_ratio = float(fg_ratio)
        self.attempts = int(attempts)

        # Precompute per-volume shape and valid Z-bands.
        self.meta = []
        for cpath, ppath in self.pairs:
            ct_v = nib.load(cpath).get_fdata().astype(np.float32)
            pt_v = nib.load(ppath).get_fdata().astype(np.float32)
            if ct_v.shape != pt_v.shape:
                raise ValueError(f"CT/PT shape mismatch: {cpath} vs {ppath}: {ct_v.shape} vs {pt_v.shape}")
            # to (D,H,W) = (Z,Y,X)
            ct_z = np.transpose(ct_v, (2,1,0))
            pt_z = np.transpose(pt_v, (2,1,0))
            bands, valid = compute_valid_z_bands(ct_z, pt_z)
            self.meta.append({
                "cpath": cpath,
                "ppath": ppath,
                "shape": ct_z.shape,
                "bands": bands,
                "valid": valid,
            })
            del ct_v, pt_v, ct_z, pt_z

        self._cache = {}  # vol_idx -> (ct_vol(DHW), pt_vol(DHW))

    def __len__(self):
        return self.num_samples

    def _get_volumes(self, vol_idx):
        if vol_idx not in self._cache:
            cpath = self.meta[vol_idx]["cpath"]
            ppath = self.meta[vol_idx]["ppath"]
            ct = nib.load(cpath).get_fdata().astype(np.float32)
            pt = nib.load(ppath).get_fdata().astype(np.float32)
            ct = np.transpose(ct, (2,1,0))
            pt = np.transpose(pt, (2,1,0))
            self._cache = {vol_idx: (ct, pt)}  # keep small
        return self._cache[vol_idx]

    @staticmethod
    def _rand_start(full, size):
        if full <= size:
            return 0
        return random.randint(0, full - size)

    def _sample_z0_from_bands(self, bands, pd, D):
        """
        Choose z0 so that [z0, z0+pd) lies within a valid band if possible.
        If no band has length >= pd, pick the largest band start (will pad/crop later).
        """
        candidates = []
        for (z0, z1) in bands:
            L = z1 - z0
            if L >= pd:
                candidates.extend(range(z0, z1 - pd + 1))
        if candidates:
            return random.choice(candidates)

        if not bands:
            return 0
        weights = [b[1]-b[0] for b in bands]
        idx = random.choices(range(len(bands)), weights=weights, k=1)[0]
        return bands[idx][0]

    def _crop_or_pad(self, vol, z0, y0, x0):
        pd,ph,pw = self.patch_size
        D,H,W = vol.shape
        z1,y1,x1 = z0+pd, y0+ph, x0+pw
        iz0, iy0, ix0 = max(0, z0), max(0, y0), max(0, x0)
        iz1, iy1, ix1 = min(D, z1), min(H, y1), min(W, x1)
        patch = vol[iz0:iz1, iy0:iy1, ix0:ix1]
        pad_z0 = iz0 - z0
        pad_y0 = iy0 - y0
        pad_x0 = ix0 - x0
        pad_z1 = z1 - iz1
        pad_y1 = y1 - iy1
        pad_x1 = x1 - ix1
        if any(v != 0 for v in (pad_z0,pad_z1,pad_y0,pad_y1,pad_x0,pad_x1)):
            patch = np.pad(patch,
                           ((pad_z0, pad_z1),(pad_y0, pad_y1),(pad_x0, pad_x1)),
                           mode="edge")
        return patch

    def __getitem__(self, i):
        vol_idx = random.randrange(len(self.meta))
        meta = self.meta[vol_idx]
        D,H,W = meta["shape"]
        pd,ph,pw = self.patch_size

        z0 = self._sample_z0_from_bands(meta["bands"], pd, D)
        y0 = self._rand_start(H, ph)
        x0 = self._rand_start(W, pw)

        ct, pt = self._get_volumes(vol_idx)
        ct_blk = self._crop_or_pad(ct, z0,y0,x0)
        pt_blk = self._crop_or_pad(pt, z0,y0,x0)

        want_fg = (random.random() < self.fg_ratio)
        tries = 0
        while want_fg and tries < self.attempts and not has_foreground(ct_blk, pt_blk):
            z0 = self._sample_z0_from_bands(meta["bands"], pd, D)
            y0 = self._rand_start(H, ph)
            x0 = self._rand_start(W, pw)
            ct_blk = self._crop_or_pad(ct, z0,y0,x0)
            pt_blk = self._crop_or_pad(pt, z0,y0,x0)
            tries += 1

        ct_blk = clip_scale_ct(ct_blk)
        pt_blk = clip_scale_pet(pt_blk)

        ct_t = torch.from_numpy(ct_blk[None, ...]).float()
        pt_t = torch.from_numpy(pt_blk[None, ...]).float()
        return ct_t, pt_t

# ------------------------
# 3D Models
# ------------------------

def conv3d_k(in_c, out_c, k=3, s=1, p=None, bias=True, norm=True, activation=None):
    if p is None:
        p = k // 2
    layers = [nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)]
    if norm:
        layers.append(nn.InstanceNorm3d(out_c))
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "lrelu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation for 3D channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class ResBlockPP3D(nn.Module):
    """ResNet++ residual block: Conv-IN-ReLU-Conv-IN + SE attention (3D)."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = conv3d_k(ch, ch, k=3, s=1, norm=True, activation="relu")
        self.conv2 = conv3d_k(ch, ch, k=3, s=1, norm=True, activation=None)
        self.se = SEBlock3D(ch, reduction=16)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.se(out)
        out = out + x
        return self.act(out)

class ResNetPPGenerator3D(nn.Module):
    """
    3D ResNet++ generator for 1->1 translation.
      7x7x7 conv -> down x2 -> N residual blocks (SE) -> up x2 -> 7x7x7 + Sigmoid
    """
    def __init__(self, in_c=1, out_c=1, base=32, n_blocks=6):
        super().__init__()
        self.c1 = conv3d_k(in_c, base, k=7, s=1, norm=True, activation="relu")
        self.d1 = nn.Sequential(
            nn.Conv3d(base, base*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(base*2),
            nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Conv3d(base*2, base*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(base*4),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlockPP3D(base*4) for _ in range(n_blocks)])
        self.u1 = nn.Sequential(
            nn.ConvTranspose3d(base*4, base*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(base*2),
            nn.ReLU(inplace=True),
        )
        self.u2 = nn.Sequential(
            nn.ConvTranspose3d(base*2, base, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(base),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Sequential(
            conv3d_k(base, out_c, k=7, s=1, norm=False, activation=None),
            nn.Sigmoid()
        )

    @staticmethod
    def _pad_to_multiple(x, mult=4, mode="replicate"):
        """
        Pad D/H/W to a multiple of mult. Returns padded tensor and crop sizes.
        F.pad format for 5D is (W_left,W_right, H_left,H_right, D_left,D_right).
        """
        b, c, d, h, w = x.shape
        pad_d = (mult - (d % mult)) % mult
        pad_h = (mult - (h % mult)) % mult
        pad_w = (mult - (w % mult)) % mult
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return x, (0,0,0,0,0,0)
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode=mode)
        return x, (0, pad_d, 0, pad_h, 0, pad_w)

    @staticmethod
    def _crop(x, pads):
        _, pd, _, ph, _, pw = pads
        d = x.size(-3) - pd if pd > 0 else x.size(-3)
        h = x.size(-2) - ph if ph > 0 else x.size(-2)
        w = x.size(-1) - pw if pw > 0 else x.size(-1)
        return x[..., :d, :h, :w]

    def forward(self, x):
        xin, pads = self._pad_to_multiple(x, mult=4, mode="replicate")
        y = self.c1(xin)
        y = self.d1(y)
        y = self.d2(y)
        y = self.res(y)
        y = self.u1(y)
        y = self.u2(y)
        y = self.outc(y)
        y = self._crop(y, pads)
        dd = x.size(-3) - y.size(-3)
        dh = x.size(-2) - y.size(-2)
        dw = x.size(-1) - y.size(-1)
        if dd != 0 or dh != 0 or dw != 0:
            y = F.pad(y, (0, max(0,dw), 0, max(0,dh), 0, max(0,dd)))
            y = y[..., :x.size(-3), :x.size(-2), :x.size(-1)]
        return y

class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, k=4, s=2, p=1, norm=True, activation="lrelu"):
        super().__init__()
        layers = [nn.Conv3d(in_c, out_c, k, s, p, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm3d(out_c))
        if activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator (shallower): [b] -> [2b] -> [4b s=1] -> out."""
    def __init__(self, in_c=2, base=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock3D(in_c, base, norm=False),          # down
            ConvBlock3D(base, base*2),                    # down
            ConvBlock3D(base*2, base*4, s=1),             # keep stride 1
            nn.Conv3d(base*4, 1, kernel_size=4, stride=1, padding=1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------
# Training
# ------------------------

def save_checkpoint(gen, disc, opt_g, opt_d, path, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "gen": gen.state_dict(),
        "disc": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, path)

def load_checkpoint(gen, disc, opt_g, opt_d, path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    gen.load_state_dict(ckpt["gen"])
    if disc is not None and "disc" in ckpt:
        disc.load_state_dict(ckpt["disc"])
    if opt_g is not None and "opt_g" in ckpt:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and "opt_d" in ckpt:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt.get("epoch", None)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairedPatches3D(
        args.train_dir,
        patch_size=tuple(args.patch_size),
        num_patches_per_epoch=args.num_patches_per_epoch,
        fg_ratio=args.fg_ratio,
        attempts=args.fg_attempts
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    gen = ResNetPPGenerator3D(in_c=1, out_c=1, base=args.gen_base, n_blocks=args.gen_blocks).to(device)
    disc = PatchDiscriminator3D(in_c=2, base=args.disc_base).to(device)

    adv_criterion = nn.BCEWithLogitsLoss()
    l1_criterion = nn.L1Loss()
    lambda_L1 = args.lambda_l1

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 1
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Resuming from {args.checkpoint}")
        epoch_saved = load_checkpoint(gen, disc, opt_g, opt_d, args.checkpoint, map_location=device)
        if epoch_saved is not None:
            start_epoch = epoch_saved + 1

    gen.train()
    disc.train()

    iters_per_epoch = len(dl)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        running_g, running_d = 0.0, 0.0

        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {epoch}/{args.epochs}", ncols=0)
        for it, (ct, pt) in enumerate(pbar, 1):
            ct = ct.to(device)  # (B,1,D,H,W) in [0,1]
            pt = pt.to(device)  # (B,1,D,H,W) in [0,1]

            # ---- Discriminator ----
            with torch.no_grad():
                fake_pt = gen(ct)
            real_in = torch.cat([ct, pt], dim=1)
            fake_in = torch.cat([ct, fake_pt], dim=1)
            d_real = disc(real_in)
            d_fake = disc(fake_in)
            d_loss = (adv_criterion(d_real, torch.ones_like(d_real)) +
                      adv_criterion(d_fake, torch.zeros_like(d_fake))) * 0.5
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ---- Generator ----
            fake_pt = gen(ct)
            d_fake_for_g = disc(torch.cat([ct, fake_pt], dim=1))
            g_adv = adv_criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_l1  = l1_criterion(fake_pt, pt) * lambda_L1
            g_loss = g_adv + g_l1
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.1f}s | "
              f"D={running_d/iters_per_epoch:.4f} | G={running_g/iters_per_epoch:.4f}")

        os.makedirs(args.checkpoints_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoints_dir, f"epoch_{epoch}.pt")
        save_checkpoint(gen, disc, opt_g, opt_d, ckpt_path, epoch)

# ------------------------
# 3D Inference (center-crop stitching)
# ------------------------

@torch.no_grad()
def infer_on_folder(args):
    """
    3D sliding-window inference with:
      - extra context margin around each patch,
      - central crop accumulation only,
      - smooth (Tukey) weights on the central crop,
      - edge padding to multiples of 4 in D/H/W,
      - overlap averaging to reduce seams,
      - preservation of input affine/header.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = ResNetPPGenerator3D(in_c=1, out_c=1, base=args.gen_base, n_blocks=args.gen_blocks).to(device)

    # Resolve checkpoint
    if args.checkpoint is None or not os.path.isfile(args.checkpoint):
        ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "epoch_*.pt")))
        if not ckpts:
            raise FileNotFoundError("No checkpoint specified and none found in --checkpoints_dir")
        args.checkpoint = ckpts[-1]
        print(f"Using latest checkpoint: {args.checkpoint}")

    load_checkpoint(gen, disc=None, opt_g=None, opt_d=None, path=args.checkpoint, map_location=device)
    gen.eval()

    ct_dir = os.path.join(args.inference_dir, "CT")
    assert os.path.isdir(ct_dir), f"Missing inference CT dir: {ct_dir}"
    exts = ("*.nii", "*.nii.gz")
    ct_paths = sorted([p for ext in exts for p in glob.glob(os.path.join(ct_dir, ext))])
    if not ct_paths:
        raise RuntimeError("No CT NIfTI files found for inference.")

    os.makedirs(args.output_dir, exist_ok=True)

    pd, ph, pw = args.patch_size
    # strides based on requested overlap of the *central* crop
    sd = max(1, int(pd * (1 - args.overlap)))
    sh = max(1, int(ph * (1 - args.overlap)))
    sw = max(1, int(pw * (1 - args.overlap)))

    # ---- weighting for the central crop only (smooth taper) ----
    def tukey_1d(L, alpha=0.5):
        n = np.arange(L, dtype=np.float32)
        w = np.ones(L, dtype=np.float32)
        if alpha <= 0:
            return w
        if alpha >= 1:
            # Hann
            return 0.5 * (1 - np.cos(2*np.pi*n/(L-1))) if L > 1 else w
        edge = int(np.floor(alpha*(L-1)/2.0))
        if edge > 0:
            i = np.arange(edge, dtype=np.float32)
            w[:edge] = 0.5*(1 + np.cos(np.pi*(2*i/(alpha*(L-1)) - 1)))
            w[-edge:] = w[edge-1::-1]
        return w

    def make_center_weight(pd, ph, pw, alpha=0.5):
        wz = tukey_1d(pd, alpha)[:, None, None]
        wy = tukey_1d(ph, alpha)[None, :, None]
        wx = tukey_1d(pw, alpha)[None, None, :]
        w = wz * wy * wx
        w /= max(1e-8, w.max())
        return w.astype(np.float32)

    center_weight = make_center_weight(pd, ph, pw, alpha=0.5)

    def pad_to_multiple_edge_3d(arr3d, mult=4):
        D,H,W = arr3d.shape
        pad_d = (mult - (D % mult)) % mult
        pad_h = (mult - (H % mult)) % mult
        pad_w = (mult - (W % mult)) % mult
        if pad_d==pad_h==pad_w==0:
            return arr3d, (0,0,0,0,0,0)  # z0,z1,y0,y1,x0,x1
        padded = np.pad(arr3d,
                        ((0, pad_d), (0, pad_h), (0, pad_w)),
                        mode="edge")
        return padded, (0, pad_d, 0, pad_h, 0, pad_w)

    m = int(max(0, args.context_margin))  # context margin per side

    for cpath in ct_paths:
        print(f"Inferring sPET for {os.path.basename(cpath)} ...")
        ct_img = nib.load(cpath)
        ct = ct_img.get_fdata().astype(np.float32)
        if ct.ndim != 3:
            raise ValueError(f"Expected 3D CT volume: {cpath}")

        # Reorder to (D,H,W)
        ct = np.transpose(ct, (2,1,0))

        # Scale and pad to multiple of 4
        ct01 = clip_scale_ct(ct)
        ct01_pad, pads = pad_to_multiple_edge_3d(ct01, 4)
        Dp, Hp, Wp = ct01_pad.shape

        out_sum = np.zeros_like(ct01_pad, dtype=np.float32)
        w_sum   = np.zeros_like(ct01_pad, dtype=np.float32)

        # start indices to cover edges, then force last start to hit the final border
        z_starts = list(range(0, max(1, Dp - pd + 1), sd)) or [0]
        y_starts = list(range(0, max(1, Hp - ph + 1), sh)) or [0]
        x_starts = list(range(0, max(1, Wp - pw + 1), sw)) or [0]
        if z_starts[-1] != Dp - pd: z_starts.append(max(0, Dp - pd))
        if y_starts[-1] != Hp - ph: y_starts.append(max(0, Hp - ph))
        if x_starts[-1] != Wp - pw: x_starts.append(max(0, Wp - pw))

        # sliding with context + center-crop accumulation
        for z in tqdm(z_starts, desc="  sliding Z", ncols=0):
            for y in y_starts:
                for x in x_starts:
                    # region WITH CONTEXT (clip at global borders)
                    z0_in = max(0, z - m);        z1_in = min(Dp, z + pd + m)
                    y0_in = max(0, y - m);        y1_in = min(Hp, y + ph + m)
                    x0_in = max(0, x - m);        x1_in = min(Wp, x + pw + m)

                    in_blk = ct01_pad[z0_in:z1_in, y0_in:y1_in, x0_in:x1_in]

                    # ensure size (pd+2m, ph+2m, pw+2m) by padding at the far ends if needed
                    need_d = pd + 2*m - in_blk.shape[0]
                    need_h = ph + 2*m - in_blk.shape[1]
                    need_w = pw + 2*m - in_blk.shape[2]
                    if need_d > 0 or need_h > 0 or need_w > 0:
                        in_blk = np.pad(
                            in_blk,
                            ((0, max(0, need_d)), (0, max(0, need_h)), (0, max(0, need_w))),
                            mode="edge"
                        )

                    # run generator on context block
                    ct_t = torch.from_numpy(in_blk[None, None, ...]).float().to(device)
                    pred01_full = gen(ct_t).cpu().numpy()[0, 0]

                    # take CENTRAL (pd,ph,pw) crop (ignore edge predictions)
                    zc0 = m; yc0 = m; xc0 = m
                    zc1 = zc0 + pd; yc1 = yc0 + ph; xc1 = xc0 + pw
                    pred01 = pred01_full[zc0:zc1, yc0:yc1, xc0:xc1]

                    # accumulate only the center region into output with smooth weights
                    out_sum[z:z+pd, y:y+ph, x:x+pw] += pred01 * center_weight
                    w_sum  [z:z+pd, y:y+ph, x:x+pw] += center_weight

        # Avoid div-by-zero
        w_sum[w_sum == 0] = 1.0
        pred01_full = out_sum / w_sum

        # Crop back to original size and reorder to (X,Y,Z) for saving
        z0,z1,y0,y1,x0,x1 = pads
        pred01 = pred01_full[:pred01_full.shape[0]-z1 if z1>0 else None,
                             :pred01_full.shape[1]-y1 if y1>0 else None,
                             :pred01_full.shape[2]-x1 if x1>0 else None]

        spet = descale_pet_from01(pred01.astype(np.float32))
        spet = np.transpose(spet, (2,1,0))  # back to (X,Y,Z)

        out_img = nib.Nifti1Image(spet.astype(np.float32), affine=ct_img.affine, header=ct_img.header)
        out_path = os.path.join(args.output_dir, os.path.basename(cpath))
        nib.save(out_img, out_path)
        print(f"Saved: {out_path}")

# ------------------------
# Main / CLI
# ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CT→sPET training & inference (pix2pix, NIfTI, **3D**) with robust stitching.")
    p.add_argument("--mode", type=str, choices=["train", "test"], default="train")

    # Training
    p.add_argument("--train_dir", type=str, default="./train", help="Folder with CT/ and PT/ subfolders")
    p.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size (# of 3D patches)")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to .pt to resume or for inference")

    # 3D patching
    p.add_argument("--patch_size", type=int, nargs=3, default=[64,64,64], metavar=("D","H","W"))
    p.add_argument("--num_patches_per_epoch", type=int, default=4000)
    p.add_argument("--fg_ratio", type=float, default=0.7, help="Probability to sample a foreground patch")
    p.add_argument("--fg_attempts", type=int, default=30, help="Max attempts to find FG patch")

    # Model widths/blocks & losses
    p.add_argument("--gen_base", type=int, default=32)
    p.add_argument("--gen_blocks", type=int, default=6)
    p.add_argument("--disc_base", type=int, default=32)
    p.add_argument("--lambda_l1", type=float, default=100.0)

    # Inference
    p.add_argument("--inference_dir", type=str, default="./inference", help="Folder with CT/ subfolder")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap fraction for the CENTRAL crop [0..0.9]")
    p.add_argument("--context_margin", type=int, default=12, help="Extra voxels per side to include as context during inference")

    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    if args.mode == "train":
        print(f"Training 3D with epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
              f"patch={tuple(args.patch_size)} (slice prefilter enabled)")
        train(args)
    else:
        print("Running 3D inference with center-crop stitching...")
        infer_on_folder(args)

if __name__ == "__main__":
    main()

