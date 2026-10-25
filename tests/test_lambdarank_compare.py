import json
from src.lambdarank_compare import export_comparison


def test_comparison_orders_by_validation_and_pairs_own_baseline(tmp_path):
    def variant(name, kind, valid, test):
        return dict(variant=name, type=kind,
                    validation={'ndcg@10': valid}, test={'ndcg@10': test})
    analysis = dict(
        variants=[variant('frozen_bpr', 'frozen', .3, .9),
                  variant('lambdarank_bpr', 'lambdarank', .4, .8),
                  variant('rrf', 'unsupervised_fusion', .2, .95)],
        best_frozen_reference_on_validation='frozen_bpr',
        selected_champion_on_validation='lambdarank_bpr',
        measured_test_delta_vs_frozen={'delta_ndcg': -.1})
    (tmp_path / 'top_models_analysis.json').write_text(json.dumps(analysis))
    rows = export_comparison(tmp_path)
    assert rows[0]['variant'] == 'lambdarank_bpr'
    assert abs(rows[0]['test_ndcg_delta_vs_own_base'] + .1) < 1e-9
    assert rows[-1]['test_ndcg_delta_vs_own_base'] is None
    assert (tmp_path / 'comparison.csv').is_file()
    assert 'Selected champion: lambdarank_bpr' in (tmp_path / 'comparison.md').read_text()
