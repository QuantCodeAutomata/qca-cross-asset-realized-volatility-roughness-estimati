"""Publication coverage is evidence, not a declaration of empirical replication."""
import json
from src.replication.coverage import collect
from src.replication.monte_carlo import build_design


def test_coverage_empty_has_every_table_and_figure():
    rows = collect()
    assert {x['item'] for x in rows} == {f'{kind}{i}' for kind in ('table', 'figure') for i in range(1,8)}
    assert all(x['status']=='requires_source_or_separate_run' for x in rows)
    assert all(not x['full_empirical_replication'] for x in rows)


def test_coverage_smoke_is_not_paper_or_empirical(tmp_path):
    scenarios = [{'scenario_id': s.id, 'n_completed':s.n_paths,
                  'path_status_counts':{'completed':s.n_paths}}
                 for s in build_design('smoke', [3,4,5,6])]
    path = tmp_path/'mc.json'
    path.write_text(json.dumps({'profile':'smoke','scenarios':scenarios}))
    table2 = tmp_path/'table2.json'
    table2.write_text(json.dumps({'cells':48, 'match_published_four_decimal_rounding':34}))
    rows = {x['item']:x for x in collect(monte_carlo=path, table2=table2)}
    for n in (3,4,5,6):
        assert rows[f'table{n}']['verified_on_fixture']
        assert not rows[f'table{n}']['executed_numerically']
    assert rows['table2']['executed_numerically']
    assert rows['table2']['printed_cells_not_matching_rounding'] == 14
    assert all(not x['executed_on_supplied_market_data'] for x in rows.values())


def test_figure2_fixture_evidence_does_not_become_full_mc(tmp_path):
    path = tmp_path/'figure2.json'
    path.write_text(json.dumps({'data_kind':'synthetic_fixture','H':.5,'kappas':[0.,.01,.035],
        'n_days':96,'n_paths_each':1,'n_completed':3,'moment_output_sha256':'0'*64}))
    row = next(x for x in collect(figure2_fixture=path) if x['item']=='figure2')
    assert row['verified_on_fixture'] and not row['executed_numerically']
    assert not row['executed_on_supplied_market_data']
