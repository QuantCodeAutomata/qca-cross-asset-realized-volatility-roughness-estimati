"""Optional local renderers for paper Figure 1--7 data products.

These are labeled reconstructions from supplied data, not copies of published
figures. Weak identification is kept visible. No data are loaded from a network.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, private_output, sha256_file


def render_empirical(result, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = private_output(output_dir)
    frames = result["frames"]
    statuses = {}

    def save(fig, name):
        fig.tight_layout()
        path = output/(name+".png")
        fig.savefig(path,dpi=150)
        plt.close(fig)
        statuses[name] = {"status":"rendered_from_supplied_fixture_or_data", "sha256":sha256_file(path)}

    if "figure1_moments" in frames:
        for index,(instrument,curve) in enumerate(frames["figure1_moments"].groupby("instrument_id")):
            fig,ax = plt.subplots(figsize=(7,4.5))
            for overlapping,group in curve.groupby("overlapping"):
                group = group[np.isfinite(group.m2) & (group.m2>0)]
                ax.plot(np.log(group.lag),np.log(group.m2),".",label="overlap="+str(overlapping))
                fits = frames["figure1_fits"]
                fits = fits[(fits.instrument_id==instrument)&(fits.overlapping==overlapping)]
                for _,row in fits.iterrows():
                    if np.isfinite(row.slope):
                        x = np.log(np.arange(row.lag_min,row.lag_max+1,dtype=float))
                        ax.plot(x,row.intercept+row.slope*x,label=f"{row.lag_min:g}-{row.lag_max:g}, H={row.H_hat:.3f}, R²={row.r_squared:.3f}")
            ax.set(xlabel="log trading-day lag",ylabel="log second moment",title="Figure 1 reconstruction: "+str(instrument))
            ax.legend(fontsize=7)
            save(fig,f"figure1_{index:02d}")
    if "equities_H" in frames and "equities_universe" in frames:
        h = frames["equities_H"]
        h = h[(h.estimator=="tsrv")&(h.lag_min==1)&(h.lag_max==10)&(~h.overlapping)]
        h = h.merge(frames["equities_universe"],on="instrument_id",validate="many_to_one")
        fig,ax = plt.subplots(figsize=(7,4.5))
        for flag,label in (("base_eligible","base universe"),("quality_eligible","quality subset")):
            values = h.loc[h[flag],"H_hat"].dropna()
            if len(values):
                ax.hist(values,bins=30,histtype="step",label=f"{label} (n={len(values)})")
                ax.axvline(values.median(),linestyle="--")
        ax.set(xlabel="TSRV H, non-overlapping lags 1–10",ylabel="assets",title="Figure 3 reconstruction")
        if ax.get_legend_handles_labels()[0]: ax.legend()
        save(fig,"figure3")
    if "option_daily_H" in frames:
        for index,(underlying,group) in enumerate(frames["option_daily_H"].groupby("underlying_id")):
            fig,ax = plt.subplots(figsize=(8,4))
            group = group.sort_values("date")
            ax.plot(pd.to_datetime(group.date),group.H_hat,".",label="daily H (including weak fits)")
            ax.plot(pd.to_datetime(group.date),group.H_hat.rolling(21,min_periods=1).median(),label="21-observation-day median")
            ax.set(xlabel="date",ylabel="implied H",title="Figure 4 reconstruction: "+str(underlying))
            ax.legend()
            save(fig,f"figure4_{index:02d}")
    if "futures_H" in frames:
        h = frames["futures_H"]
        h = h[(h.estimator=="tsrv")&(h.lag_min==1)&(h.lag_max==10)&(~h.overlapping)]
        corrections = frames["futures_corrections"]
        corrections = corrections[~corrections.overlapping]
        h = h.merge(corrections[["instrument_id","H_corrected_b","optimizer_success","correction_valid_10"]],on="instrument_id",how="left")
        fig,ax = plt.subplots(figsize=(max(7,len(h)*.25),4.5))
        positions = np.arange(len(h))
        ax.plot(positions,h.H_hat,"o",label="raw TSRV")
        valid = h.optimizer_success.fillna(False)&h.correction_valid_10.fillna(False)
        ax.plot(positions[valid],h.loc[valid,"H_corrected_b"],"^",fillstyle="none",label="corrected RK (valid fits)")
        ax.set_xticks(positions,h.instrument_id,rotation=90)
        ax.set(ylabel="H, non-overlapping lags 1–10",title="Figure 5 reconstruction")
        ax.legend()
        save(fig,"figure5")
    if "realized_implied_match" in frames:
        data = frames["realized_implied_match"]
        for number,xcolumn,valid_column,label in ((6,"H_realized_TSRV","raw_pair_available","raw TSRV H"),
                                                   (7,"H_realized_RK_corrected","corrected_pair_available","noise-corrected RK H")):
            subset = data[data[valid_column]]
            if subset.empty:
                kind = "raw" if number == 6 else "corrected"
                statuses[f"figure{number}"] = {"status":f"no_valid_{kind}_pairs", "n_pairs":0}
                continue
            fig,ax = plt.subplots(figsize=(6,5))
            for identified,group in subset.groupby("identified",dropna=False):
                ax.plot(group[xcolumn],group.H_hat_IV,"o",fillstyle="full" if identified is True or identified==True else "none",
                        label="identified" if identified is True or identified==True else "weak/unidentified")
            if len(subset):
                lo = float(np.nanmin(subset[[xcolumn,"H_hat_IV"]].values))
                hi = float(np.nanmax(subset[[xcolumn,"H_hat_IV"]].values))
                ax.plot([lo,hi],[lo,hi],"--",label="H implied = H realized")
                ax.legend()
            ax.set(xlabel=label,ylabel="pooled implied H",title=f"Figure {number} reconstruction")
            save(fig,f"figure{number}")
    for number in (1,3,4,5,6,7):
        if not any(key==f"figure{number}" or key.startswith(f"figure{number}_") for key in statuses):
            statuses[f"figure{number}"] = {"status":"requires_source_or_selection"}
    atomic_json(output/"figure_status.json",statuses)
    return statuses


def render_figure2(moment_frame, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    required = {"scenario_id","estimator","lag","mean_moment"}
    if not required.issubset(moment_frame) or moment_frame.empty:
        raise ValueError("Figure 2 requires smooth-truth simulation moment products")
    output = private_output(output_dir)
    fig,ax = plt.subplots(figsize=(7,4.5))
    for key,group in moment_frame.groupby(["scenario_id","estimator"]):
        group = group[np.isfinite(group.mean_moment)&(group.mean_moment>0)]
        ax.plot(np.log(group.lag),np.log(group.mean_moment),label=" / ".join(key))
    ax.set(xlabel="log trading-day lag",ylabel="log ensemble mean second moment",title="Figure 2 reconstruction, H=0.5")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path=output/"figure2.png"
    fig.savefig(path,dpi=150); plt.close(fig)
    return sha256_file(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-output",type=Path)
    parser.add_argument("--figure2-moments",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if bool(args.empirical_output)==bool(args.figure2_moments):
        parser.error("select exactly one source of figure data")
    if args.figure2_moments:
        print(render_figure2(pd.read_csv(args.figure2_moments),args.output_dir))
    else:
        # Only explicit data product basenames are read, not arbitrary private CSVs.
        names=("figure1_moments","figure1_fits","equities_H","equities_universe","option_daily_H",
               "futures_H","futures_corrections","realized_implied_match")
        frames={name:pd.read_csv(args.empirical_output/(name+".csv")) for name in names if (args.empirical_output/(name+".csv")).exists()}
        print(render_empirical({"frames":frames},args.output_dir))


if __name__=="__main__":
    main()
