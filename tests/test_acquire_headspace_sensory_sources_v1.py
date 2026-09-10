from __future__ import annotations

import hashlib

from scripts.acquire_headspace_sensory_sources_v1 import pubchem_supplement


def test_pubchem_supplement_reproduces_reviewed_bytes() -> None:
    payload = pubchem_supplement(
        {
            "CID": 1549778,
            "MolecularWeight": "194.31",
            "SMILES": "CC(=CCC/C(=C/CCC(=O)C)/C)C",
            "ConnectivitySMILES": "CC(=CCCC(=CCCC(=O)C)C)C",
            "InChIKey": "HNZUNIKWNYHEJJ-FMIVXFBMSA-N",
            "IUPACName": "(5E)-6,10-dimethylundeca-5,9-dien-2-one",
        }
    )
    assert len(payload) == 587
    assert hashlib.sha256(payload).hexdigest() == (
        "87f34f2c7758de7aeded1a6db786c0ba7a2778c0899850294fcec294bef1609c"
    )
