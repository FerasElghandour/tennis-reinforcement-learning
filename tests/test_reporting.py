import json
from reporting import publish_report


def test_report_escapes_embedded_script_and_records_input_hash(tmp_path):
    source=tmp_path/'input.csv';source.write_text('x,y\n1,2\n')
    spec={'title':'A safe report','subtitle':'Check','category':'Research','findings':['</script><script>alert(1)</script>'], 'views':[], 'methodology':[], 'notes':'Aggregate data'}
    output=tmp_path/'report'
    publish_report(output,{'rows':1},spec,[source])
    text=(output/'index.html').read_text()
    assert '</script><script>alert(1)</script>' not in text
    assert '\\u003c/script' in text
    data=json.loads((output/'metrics.json').read_text())
    assert len(data['provenance']['inputs'][0]['sha256'])==64
    assert data['provenance']['inputs'][0]['name']=='input.csv'
    assert str(tmp_path) not in text
