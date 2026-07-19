#!/usr/bin/env python
"""NIRSpec 6927 target screen: catch clipped-saturated-star selections and
targets with no counterpart in the current catalog.

Tests per target:
  T1 SATSTAR: match vs consolidated satstar catalogs (0.5"). If the CSV mag in
     that band is >=2 mag FAINTER than the satstar-implied mag, the selection
     photometry is a clipped-saturation artifact -> FAIL (unless REFERENCE,
     where bright is expected -> note only, but mag mismatch still WARNs).
  T2 EXISTS: counterpart in current m8_dedup within 0.3" with finite flux in
     at least one CSV selection band -> else FAIL.
  T3 FLAGS: counterpart replaced_saturated / forced_filled / low_fit_quality
     in a selection band -> WARN (FAIL if replaced_saturated and CSV >=2 mag
     fainter than catalog).
  T4 PAIR-SANITY: current-catalog |F405N-F410M| > 1 mag -> WARN (near-
     degenerate pair; big color = corrupt photometry).

Output: screen_results.csv + console summary.
"""
import numpy as np
import pandas as pd
import warnings
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path('/blue/adamginsburg/adamginsburg/jwst/brick')
CSV = BASE / 'nirspec_6927' / 'all_pointings_sources_20260715.csv'
CAT = '/orange/adamginsburg/jwst/brick/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
SATDIR = BASE / 'catalogs'
IMGROOT = Path('/orange/adamginsburg/jwst/brick')

ZP = {'f115w': 1746.1179, 'f182m': 844.9433, 'f187n': 794.8482,
      'f200w': 757.6538, 'f212n': 674.8317, 'f356w': 271.3926,
      'f405n': 206.9694, 'f410m': 208.7505, 'f444w': 184.1022,
      'f466n': 157.7731}
BANDS = list(ZP)
IMG = {b: IMGROOT / b.upper() / 'pipeline' /
       (f'jw01182-o004_t001_nircam_clear-{b}-merged_i2d.fits'
        if b in ('f115w', 'f200w', 'f356w', 'f444w') else
        f'jw02221-o001_t001_nircam_clear-{b}-merged_i2d.fits')
       for b in BANDS}

targets = pd.read_csv(CSV)
targets.columns = [c.strip('# ') for c in targets.columns]
tsc = SkyCoord(targets['RA'].values * u.deg, targets['Dec'].values * u.deg)
print(f'{len(targets)} targets')


def csv_mag(row, b):
    col = b.upper()
    if col not in row.index:
        return np.nan
    v = float(row[col])
    return v if np.isfinite(v) and v > -90 else np.nan


# satstar catalogs + implied mags
sat = {}
for b in BANDS:
    p = SATDIR / f'{b}_consolidated_satstar_catalog.fits'
    if not p.exists():
        continue
    c = Table.read(p)
    pixar = None
    if IMG[b].exists():
        with fits.open(IMG[b]) as h:
            pixar = h['SCI'].header.get('PIXAR_SR')
    if pixar is None:
        continue
    fjy = np.asarray(c['flux_fit'], float) * pixar * 1e6
    with np.errstate(invalid='ignore', divide='ignore'):
        mag = -2.5 * np.log10(fjy / ZP[b])
    sat[b] = (SkyCoord(c['skycoord_fit'].ra.deg * u.deg,
                       c['skycoord_fit'].dec.deg * u.deg), mag)
    print(f'  satstar {b}: {len(c)}')

# current catalog (box around brick)
with fits.open(CAT, memmap=True) as h:
    d = h[1].data
    ra = np.array(d['skycoord_ref.ra'], float)
    dec = np.array(d['skycoord_ref.dec'], float)
    box = (ra > 266.3) & (ra < 266.8) & (dec > -28.9) & (dec < -28.5)
    cat = Table(d[box])
csc = SkyCoord(np.array(cat['skycoord_ref.ra'], float) * u.deg,
               np.array(cat['skycoord_ref.dec'], float) * u.deg)
print(f'  m8_dedup in box: {len(cat):,}')


def cat_mag(row, b):
    c = f'flux_jy_{b}'
    if c not in cat.colnames:
        return np.nan
    try:
        f = float(row[c])
    except (TypeError, ValueError):
        return np.nan
    if not (np.isfinite(f) and f > 0):
        return np.nan
    return -2.5 * np.log10(f / ZP[b])


def flag(row, pre, b):
    c = f'{pre}_{b}'
    if c not in cat.colnames:
        return False
    try:
        return bool(row[c])
    except (TypeError, ValueError):
        return False


idx, sep, _ = tsc.match_to_catalog_sky(csc)
sat_idx = {b: tsc.match_to_catalog_sky(sat[b][0]) for b in sat}

# Self-calibrate per-band CSV-vs-catalog offsets (the CSV's 1182 wide bands
# are AB while the 2221 bands are Vega; measured medians match AB-Vega
# exactly, MAD=0.00). All mag comparisons below subtract these.
OFF = {}
for b in BANDS:
    col = b.upper()
    if col not in targets.columns:
        continue
    cm = targets[col].values.astype(float)
    cm[cm < -90] = np.nan
    diffs = []
    for i in range(len(targets)):
        if float(np.asarray(sep.arcsec).flat[i]) >= 0.3 or not np.isfinite(cm[i]):
            continue
        km = cat_mag(cat[int(np.asarray(idx).flat[i])], b)
        if np.isfinite(km):
            diffs.append(cm[i] - km)
    OFF[b] = float(np.median(diffs)) if len(diffs) >= 5 else 0.0
print('per-band CSV offsets:', {k: round(v, 2) for k, v in OFF.items()})

rows = []
for i in range(len(targets)):
    trow = targets.iloc[i]
    tid = int(trow['ID'])
    is_ref = str(trow.get('REFERENCE', '')).upper() == 'TRUE'
    selbands = [b for b in BANDS if np.isfinite(csv_mag(trow, b))]
    issues, warns, notes = [], [], []

    # T1 satstar
    for b in sat:
        si, ss = int(np.asarray(sat_idx[b][0]).flat[i]), float(np.asarray(sat_idx[b][1].arcsec).flat[i])
        if ss < 0.5:
            smag = float(sat[b][1][si])
            cm = csv_mag(trow, b)
            notes.append(f'satstar:{b}@{ss*1000:.0f}mas(m={smag:.1f})')
            if np.isfinite(cm) and np.isfinite(smag) and (cm - OFF.get(b, 0.0) - smag) >= 2.0:
                (warns if is_ref else issues).append(
                    f'T1:{b} CSV {cm:.1f} vs satstar {smag:.1f} (clipped-flux selection)')

    # T2 existence
    s_as = float(np.asarray(sep.arcsec).flat[i])
    crow = cat[int(np.asarray(idx).flat[i])] if s_as < 0.3 else None
    if crow is None:
        near_sat = any(float(np.asarray(sat_idx[b][1].arcsec).flat[i]) < 0.5 for b in sat)
        if not near_sat:
            issues.append(f'T2: no m8_dedup counterpart within 0.3" (nearest {s_as:.2f}")')
        else:
            notes.append('no normal-catalog row; satstar only')
    else:
        got = [b for b in selbands if np.isfinite(cat_mag(crow, b))]
        if selbands and not got:
            warns.append('T2b: counterpart lacks flux in all selection bands')
        # T3 flags
        for b in selbands:
            if flag(crow, 'replaced_saturated', b):
                cm, km = csv_mag(trow, b), cat_mag(crow, b)
                if np.isfinite(cm) and np.isfinite(km) and (cm - OFF.get(b, 0.0) - km) >= 2.0:
                    (warns if is_ref else issues).append(
                        f'T3:{b} replaced_saturated, CSV {cm:.1f} vs cat {km:.1f}')
                else:
                    warns.append(f'T3:{b} replaced_saturated')
            for pre in ('forced_filled', 'low_fit_quality'):
                if flag(crow, pre, b):
                    warns.append(f'T3:{b} {pre}')
        # T4 pair sanity
        m405, m410 = cat_mag(crow, 'f405n'), cat_mag(crow, 'f410m')
        if np.isfinite(m405) and np.isfinite(m410) and abs(m405 - m410) > 1.0:
            warns.append(f'T4: F405N-F410M = {m405-m410:+.1f} (corrupt pair)')
        # CSV vs catalog gross mismatch in any selection band
        for b in selbands:
            cm, km = csv_mag(trow, b), cat_mag(crow, b)
            if np.isfinite(cm) and np.isfinite(km) and (cm - OFF.get(b, 0.0) - km) >= 2.0:
                lab = f'T5:{b} CSV {cm:.1f} vs current cat {km:.1f} (off {OFF.get(b,0.0):+.2f})'
                (warns if is_ref else issues).append(lab)

    verdict = 'FAIL' if issues else ('WARN' if warns else 'PASS')
    rows.append(dict(ID=tid, Pointing=int(trow['Pointing']), REFERENCE=is_ref,
                     RA=float(trow['RA']), Dec=float(trow['Dec']),
                     sep_cat_arcsec=round(s_as, 3), verdict=verdict,
                     issues='; '.join(issues), warns='; '.join(warns),
                     notes='; '.join(notes)))

res = pd.DataFrame(rows)
out = BASE / 'nirspec_6927' / 'screen_results.csv'
res.to_csv(out, index=False)
print(f'\nwrote {out}')
print(res['verdict'].value_counts().to_string())
print('\n=== FAIL ===')
for _, r in res[res.verdict == 'FAIL'].iterrows():
    print(f"  ID {r.ID} P{r.Pointing}{' [REF]' if r.REFERENCE else ''}: {r.issues}")
print('\n=== WARN (first 40) ===')
for _, r in res[res.verdict == 'WARN'].head(40).iterrows():
    print(f"  ID {r.ID} P{r.Pointing}{' [REF]' if r.REFERENCE else ''}: {r.warns}")
