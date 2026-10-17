import hashlib
import json
from pathlib import Path


def test_published_report_matches_current_source_and_has_finite_json():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'docs/metrics.json').read_text())
    assert data['report']['views']
    for entry in data['provenance']['code']:
        assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    assert '<script type="application/json" id="payload">' in (root/'docs/index.html').read_text()
