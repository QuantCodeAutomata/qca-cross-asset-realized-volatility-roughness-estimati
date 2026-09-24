"""Combine numerical, smoke and empirical receipts without promoting a paper claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import atomic_json, sha256_file
from .empirical import COVERAGE
from .monte_carlo import build_design


def collect(*, empirical=None, monte_carlo=None, table2=None, figure2_fixture=None):
    status={item:{"item":item,"implemented":True,"verified_on_fixture":False,
        "executed_numerically":False,"executed_on_supplied_market_data":False,
        "requires_methodology_clarification":note,"evidence":[],"full_empirical_replication":False}
        for item,(_,note) in COVERAGE.items()}
    if empirical:
        receipt=json.loads(Path(empirical).read_text())
        for row in receipt["coverage"]:
            target=status[row["item"]]
            target["verified_on_fixture"] |= bool(row["checked_on_this_fixture"])
            target["executed_on_supplied_market_data"] |= bool(row["executed_on_supplied_market_data"])
            target["requires_source_or_separate_run"]=row["requires_source_or_separate_run"]
            target["evidence"].append({"kind":"empirical_receipt","sha256":sha256_file(empirical)})
    if monte_carlo:
        receipt=json.loads(Path(monte_carlo).read_text())
        profile=receipt["profile"]
        by_id={s["scenario_id"]:s for s in receipt["scenarios"]}
        for table in (3,4,5,6):
            needed=build_design(profile,[table])
            complete=all(s.id in by_id and by_id[s.id]["n_completed"]==s.n_paths for s in needed)
            healthy=complete and all(by_id[s.id]["path_status_counts"].get("failed",0)==0 for s in needed)
            target=status[f"table{table}"]
            target["verified_on_fixture"] |= bool(profile=="smoke" and healthy)
            target["executed_numerically"] |= bool(profile=="paper" and complete)
            target["n_requested_paths"]=sum(s.n_paths for s in needed)
            target["n_completed_paths"]=sum(by_id.get(s.id,{}).get("n_completed",0) for s in needed)
            target["evidence"].append({"kind":profile+"_mc_receipt","sha256":sha256_file(monte_carlo)})
        status["figure2"]["executed_numerically"]=bool(profile=="paper" and status["table3"]["executed_numerically"])
    if table2:
        receipt=json.loads(Path(table2).read_text())
        if receipt["cells"] != 48:
            raise ValueError("Table 2 receipt must cover all 48 printed cells")
        status["table2"].update(executed_numerically=True,
            printed_cells_matching_rounding=receipt["match_published_four_decimal_rounding"],
            printed_cells_not_matching_rounding=48-receipt["match_published_four_decimal_rounding"])
        status["table2"]["evidence"].append({"kind":"exact_integral_comparison","sha256":sha256_file(table2)})
    if figure2_fixture:
        receipt=json.loads(Path(figure2_fixture).read_text())
        if (receipt.get("data_kind") != "synthetic_fixture" or receipt.get("H") != .5
                or receipt.get("kappas") != [0.,.01,.035]
                or receipt.get("n_completed") != 3*receipt.get("n_paths_each",0)
                or not receipt.get("moment_output_sha256")):
            raise ValueError("invalid Figure 2 fixture receipt")
        status["figure2"]["verified_on_fixture"]=True
        status["figure2"]["evidence"].append({"kind":"smooth_truth_fixture","sha256":sha256_file(figure2_fixture)})
    for row in status.values():
        row["status"]=("executed_on_supplied_data" if row["executed_on_supplied_market_data"] else
                       "executed_numerically" if row["executed_numerically"] else
                       "verified_on_fixture" if row["verified_on_fixture"] else "requires_source_or_separate_run")
    return list(status.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-report",type=Path)
    parser.add_argument("--mc-report",type=Path)
    parser.add_argument("--table2-report",type=Path)
    parser.add_argument("--figure2-fixture-report",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    atomic_json(args.output,collect(empirical=args.empirical_report,monte_carlo=args.mc_report,table2=args.table2_report,figure2_fixture=args.figure2_fixture_report))


if __name__=="__main__":
    main()
