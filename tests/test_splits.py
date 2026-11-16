import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = next(
    (p for p in (ROOT / "results", ROOT.parent / "evidence/results") if p.is_dir()),
    None,
)
pytestmark = pytest.mark.skipif(RESULTS is None, reason="no results evidence in this repo layout")

spec = importlib.util.spec_from_file_location('audit_splits', ROOT/'tools/audit_splits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recorded_split_is_clean_and_disjoint():
    r = module.audit(ROOT/'splits/mef_scene_split.json', RESULTS/'per_scene_mef.json')
    assert r['passed']
    assert len(r['methods']) == 5
    for row in r['methods'].values():
        assert row['unique_count'] == 42
        assert {k:len(v) for k,v in row['intersections'].items()} == {'train':0,'val':0,'test':42}


def test_duplicate_results_and_overlapping_splits_fail(tmp_path):
    split = tmp_path/'split.json'; results = tmp_path/'results.json'
    split.write_text(json.dumps({'splits':{'train':['a'],'val':[],'test':[{'scene':'b'}]}}))
    results.write_text(json.dumps([{'method':'m','scene':'b'}]))
    assert module.audit(split, results)['passed']
    results.write_text(json.dumps([{'method':'m','scene':'b'}]*2))
    assert not module.audit(split, results)['passed']
    split.write_text(json.dumps({'splits':{'train':['b'],'val':[],'test':['b']}}))
    assert module.audit(split, results)['split_errors']
