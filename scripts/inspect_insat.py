"""Audit INSAT-3DR L1C ASIA_MER frames in a ZIP, and render a Meghalaya TIR1 preview.

This is an inspection utility, not a training pipeline or operational nowcast.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer

AOI = (89.5, 24.7, 93.3, 26.5)  # approximate BUFFERED inspection rectangle; replace with authoritative Meghalaya boundary

def scalar(v):
    if isinstance(v, (np.ndarray, list)):
        return scalar(v[0]) if len(v) else None
    if isinstance(v, bytes):
        return v.decode('utf-8', 'replace')
    if hasattr(v, 'item'):
        return v.item()
    return v

def clip_indices(f, bbox):
    p = f['Projection_Information'].attrs
    lon0 = float(scalar(p['longitude_of_projection_origin']))
    lat_ts = float(scalar(p['standard_parallel']))
    a = float(scalar(p['semi_major_axis']))
    b = float(scalar(p['semi_minor_axis']))
    crs = CRS.from_proj4(f'+proj=merc +lon_0={lon0} +lat_ts={lat_ts} +a={a} +b={b} +units=m +no_defs')
    toxy = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    w, s, e, n = bbox
    xx, yy = toxy.transform([w,w,e,e], [s,n,s,n])
    x = f['X'][:]; y = f['Y'][:]
    cols = np.where((x >= min(xx)) & (x <= max(xx)))[0]
    rows = np.where((y >= min(yy)) & (y <= max(yy)))[0]
    if cols.size == 0 or rows.size == 0:
        raise ValueError('Selected AOI does not intersect the satellite raster')
    return slice(int(rows.min()), int(rows.max())+1), slice(int(cols.min()),int(cols.max())+1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip',required=True, help='Path to ZIP containing original INSAT-3DR *.h5 files')
    ap.add_argument('--out',default='outputs/inspection')
    args = ap.parse_args()
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    inventory=[]; images=[]; times=[]
    with zipfile.ZipFile(args.zip) as z:
        names=sorted(n for n in z.namelist() if n.lower().endswith('.h5'))
        if not names:raise ValueError('No .h5 files found in ZIP')
        for name in names:
            with h5py.File(io.BytesIO(z.read(name)),'r') as f:
                start=scalar(f.attrs.get('Acquisition_Start_Time',b''))
                end=scalar(f.attrs.get('Acquisition_End_Time',b''))
                sat=scalar(f.attrs.get('Satellite_Name',b''))
                rs,cs=clip_indices(f, AOI)
                raw=f['IMG_TIR1'][0,rs,cs]
                lookup=f['IMG_TIR1_TEMP'][:]
                fill=int(scalar(f['IMG_TIR1'].attrs['_FillValue']))
                valid=(raw!=fill)&(raw<len(lookup))
                temp=np.full(raw.shape,np.nan,dtype=np.float32)
                temp[valid]=lookup[raw[valid].astype(np.int32)]
                inventory.append({'filename':Path(name).name,'acquisition_start_utc':start,'acquisition_end_utc':end,'satellite':sat,'product':scalar(f.attrs.get('Processing_Level',b'')),'rows':f['IMG_TIR1'].shape[-2], 'cols':f['IMG_TIR1'].shape[-1], 'crop_rows':raw.shape[0],'crop_cols':raw.shape[1], 'valid_tir1_percent':round(float(valid.mean()*100),2)})
                images.append(temp);times.append(end)
    with (out/'insat_inventory.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
    intervals=[]
    for a,b in zip(inventory,inventory[1:]):
        dt=(datetime.strptime(b['acquisition_start_utc'],'%d-%b-%YT%H:%M:%S')-datetime.strptime(a['acquisition_start_utc'],'%d-%b-%YT%H:%M:%S')).total_seconds()/60
        intervals.append({'before':a['filename'],'after':b['filename'],'minutes':round(dt,2),'gap_over_45min':dt>45})
    report={'number_of_frames':len(inventory),'inspection_bbox_wsen':AOI,'dataset': 'INSAT-3DR IMAGER L1C Asia Mercator','note':'Preview is a research inspection only. Bounding rectangle is not the final Meghalaya polygon.','time_intervals':intervals}
    (out/'sample_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cols=2;rows=(len(images)+1)//2
    fig,axes=plt.subplots(rows,cols,figsize=(10,max(3,rows*3.2)),squeeze=False,constrained_layout=True)
    for ax,img,t in zip(axes.ravel(),images,times):
        im=ax.imshow(img,cmap='viridis',vmin=200,vmax=320)
        ax.set_title(f'TIR1 Kelvin | acquisition ends {t[-8:]} UTC',fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    for ax in axes.ravel()[len(images):]:ax.axis('off')
    fig.colorbar(im,ax=list(axes.ravel()),fraction=0.017,pad=0.02,label='Brightness temperature (K)')
    fig.suptitle('INSAT-3DR TIR1: approximate Meghalaya region (diagnostic preview)',fontsize=12)
    fig.savefig(out/'tir1_meghalaya_preview.png',dpi=130)
    plt.close(fig)
    print(f'Inspected {len(inventory)} frames. Wrote {out}/insat_inventory.csv, sample_report.json, tir1_meghalaya_preview.png')
    print('Intervals exceeding 45 minutes:',sum(i['gap_over_45min'] for i in intervals))

if __name__=='__main__': main()
