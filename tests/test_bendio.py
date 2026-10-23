import json
import numpy as np
import pytest
from pathlib import Path

from groot1.bendio import RequestBlob, data_offset, load_gbw, layout_from_gbw

FIXTURE = Path(__file__).parent / "fixtures" / "parity"


def _q(qtype, ids, pos, blocks, markers):
    from groot1.serialize import EncodedQuestion
    return EncodedQuestion(name="q", qtype=qtype, ids=ids, position_ids=pos, markers=markers, blocks=blocks)


def test_request_blob_layout():
    chunks = [[1, 2], [3, 4, 5]]
    q = _q(1, [2, 7, 1, 4, 9, 4, 10], [0, 1, 2, 3, 4, 3, 4], [0, 0, 0, 1, 1, 2, 2], [3, 5])  # score, 2 options
    b = RequestBlob.build(chunks, [q])
    # header: nch, clen, nq, qlen, kmax, nopt, chunk lens, per-q (type, K, L, o0)
    assert list(b[:12]) == [2, 3, 1, 7, 2, 2, 2, 3, 1, 2, 7, 0]
    body = list(b[12:])
    assert body[:6] == [1, 2, 0, 3, 4, 5]                 # padded chunk ids
    assert body[6:13] == [2, 7, 1, 4, 9, 4, 10]           # question ids
    assert body[13:20] == [0, 1, 2, 3, 4, 3, 4]           # shared option positions
    assert body[20:27] == [0, 0, 0, 1, 1, 2, 2]           # attention blocks
    assert body[27:29] == [3, 5]                          # markers
    assert len(b) == 12 + 6 + 7 * 3 + 2


def test_request_blob_write_roundtrip(tmp_path):
    b = RequestBlob.build([[7, 8, 9]], [_q(0, [2, 4, 5], [0, 1, 1], [0, 1, 2], [1, 2])])
    p = tmp_path / "req.bin"
    n = RequestBlob.write(b, p)
    assert n == b.size
    back = np.frombuffer(p.read_bytes(), dtype="<u4")
    assert (back == b).all()


@pytest.mark.skipif(not (FIXTURE / "model.gbw").exists(), reason="parity fixture missing")
def test_gbw_header_and_layout(tmp_path):
    header, data = load_gbw(FIXTURE / "model.gbw")
    meta = json.loads((FIXTURE / "meta.json").read_text())
    assert data.dtype == np.float32
    assert data_offset(FIXTURE / "model.gbw") == meta["data_offset"]
    assert data.size == meta["n_floats"] if "n_floats" in meta else True
    lay = layout_from_gbw(FIXTURE / "model.gbw", tmp_path / "layout.bend")
    text = lay.read_text()
    assert "def tree_norm" in text and "def d()" in text
    assert f"def data_offset() -> U32:\n  {meta['data_offset']}\n" in text
    committed = Path(__file__).parent.parent / "bend" / "groot" / "layout.bend"
    assert committed.read_text() == text  # the committed layout is the fixture's


def test_reference_outputs_match_fixture():
    import importlib.util

    from groot1.model import GrootDecisionModel

    spec = importlib.util.spec_from_file_location("fixture", Path(__file__).parent.parent / "scripts" / "make_parity_fixture.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    chunks, questions = fx.build_request(np.random.default_rng(0), 1024)
    from groot1.bendio import reference_outputs

    model = GrootDecisionModel.from_pretrained(FIXTURE / "ckpt").eval()
    out = reference_outputs(model, chunks, questions, fx.PAD)
    assert np.abs(out - np.load(FIXTURE / "ref.npy")).max() < 1e-5
