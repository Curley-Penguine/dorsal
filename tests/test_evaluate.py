from dorsal.evaluate import evaluate


def test_small_evaluation_runs(tmp_path):
    s = evaluate(n=5, out=str(tmp_path), n_paths=200)
    for k in ("none", "naive_rule", "dorsal"):
        assert 0 <= s[k]["shortfall_rate"] <= 1
    assert (tmp_path / "REPORT.md").exists()
