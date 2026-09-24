"""Generate toy private files and exercise all empirical adapters; no market data.

python -m tests.run_file_smoke --output-dir /private/toy-validation
The generated manifest is a fixture, never an inferred paper universe.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .panel_fixtures import calendar_fixture, dataset_config, instrument_fixture, bars_fixture, write_manifest
from .option_fixtures import price_fixture
from .test_file_panels import futures_fixture
from src.replication.common import private_output, atomic_json, sha256_file
from src.replication.empirical import run, export_approved_aggregates
from src.replication.figures import render_empirical, render_figure2
from src.replication.monte_carlo import RunConfig, Scenario, simulate_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = private_output(args.output_dir)
    days = [x.strftime('%Y-%m-%d') for x in pd.bdate_range('2024-03-01',periods=45)]
    calendar = calendar_fixture(days, minutes=10)
    instruments = instrument_fixture().iloc[[0]].copy()
    instruments['valid_from'], instruments['valid_to'] = days[0],days[-1]
    _, contracts, _, _ = futures_fixture()
    contracts['listed_from'] = days[0]
    contracts['expiry'] = ['2024-06-20','2024-09-20']
    volumes = pd.DataFrame([{'date':day,'root':'ES','contract_id':c,'volume':v}
        for i,day in enumerate(days) for c,v in [('A',100 if i<20 else 20),('B',20 if i<20 else 100)]])
    a,b = bars_fixture(calendar,level=100),bars_fixture(calendar,level=300)
    a['contract_id'],b['contract_id'] = 'A','B'
    definitions,prices,context = price_fixture(n_dates=3)
    files = {'sessions':calendar,'instruments':instruments,'equity_bars':bars_fixture(calendar),
        'second_bars':bars_fixture(calendar,bar_seconds=1),
        'futures_contracts':contracts,'futures_daily':volumes,'futures_bars':pd.concat([a,b]),
        'option_definitions':definitions,'option_prices':prices,'option_context':context}
    path = write_manifest(root/'inputs',files,
        datasets={'equity':dataset_config(),'futures':dataset_config(),'seconds':dataset_config(1)},
        selections={'seconds':['E1'],'futures_roots':['ES'],'option_underlyings':['U'],
                    'comparison_map':[{'realized_id':'E1','option_id':'U'}],'figure1_ids':['E1','ES']},
        analysis={'futures_roll_timing':'same_session','lag_windows':[[1,10],[1,40],[40,250]],'overlapping':[False,True],
                  'aggregation':{'missing_day_policy':'invalidate','edge_period_policy':'include'},
                  'option_sensitivities':[{'min_days':30},{'otm_only':True}]})
    result = run(path, root/'empirical')
    export_approved_aggregates(result, root/'approved_fixture_aggregates.json')
    render_empirical(result, root/'figures')
    records, rows = [], []
    for kappa in (0.,.01,.035):
        scenario = Scenario(3,.5,kappa,1e-4,1)
        record = simulate_path(RunConfig(),scenario,0,root/'figure2_cache')
        if record['status'] != 'completed':
            raise RuntimeError(record['failure_reason'])
        records.append({'scenario_id':scenario.id,'seeds':record['seeds']})
        for estimator,values in record['figure2_moments'].items():
            rows.extend({'scenario_id':scenario.id,'estimator':estimator,'lag':i+1,'mean_moment':v}
                        for i,v in enumerate(values))
    moments = pd.DataFrame(rows)
    moment_path = root/'figure2_fixture_moments.csv'
    moments.to_csv(moment_path,index=False)
    figure_hash = render_figure2(moments,root/'figures')
    atomic_json(root/'figure2_fixture_report.json', {'data_kind':'synthetic_fixture','H':.5,
        'kappas':[0.,.01,.035],'n_days':96,'n_paths_each':1,'n_completed':3,
        'moment_output_sha256':sha256_file(moment_path),'figure_sha256':figure_hash,
        'scenarios':records,'statistical_replication':False})
    assert result['report']['counts']['equities']['observed'] == 1
    assert result['report']['counts']['second_frequency_assets']['observed'] == 1
    assert result['report']['counts']['futures_roots']['observed'] == 1
    assert not result['report']['full_empirical_replication']
    print('Toy CSV panels and all figure products completed; no market data used.')


if __name__ == '__main__':
    main()
