from local_impact_ranking.csv_safety import sanitize_csv_cell, sanitize_csv_row


def test_sanitize_csv_cell_prefixes_formula_like_strings():
    assert sanitize_csv_cell("=SUM(1,1)") == "'=SUM(1,1)"
    assert sanitize_csv_cell("+SUM(1,1)") == "'+SUM(1,1)"
    assert sanitize_csv_cell("-SUM(1,1)") == "'-SUM(1,1)"
    assert sanitize_csv_cell("@SUM(1,1)") == "'@SUM(1,1)"
    assert sanitize_csv_cell("  =SUM(1,1)") == "'  =SUM(1,1)"
    assert sanitize_csv_cell("\tSUM(1,1)") == "'\tSUM(1,1)"
    assert sanitize_csv_cell("Ordinary paper") == "Ordinary paper"
    assert sanitize_csv_cell(-1.0) == -1.0


def test_sanitize_csv_row_preserves_keys_and_non_strings():
    assert sanitize_csv_row({"title": "=x", "score": -1.0}) == {
        "title": "'=x",
        "score": -1.0,
    }
