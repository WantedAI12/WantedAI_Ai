"""Acquire all published DOE rows from a source-specific skincare emulsion study."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8800138/fullTextXML"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                raw = response.read()
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
    root = ET.fromstring(raw)
    license_element = root.find(".//permissions/license")
    if license_element is None:
        raise ValueError("source license missing")
    license_xml = ET.tostring(license_element, encoding="unicode")
    table = root.find('.//table-wrap[@id="t0015"]')
    names = [
        "run",
        "rotor_rpm",
        "flow_l_per_h",
        "span60_wt_percent",
        "tween60_wt_percent",
        "mineral_oil_wt_percent",
        "droplet_diameter_nm",
    ]
    rows = []
    for tr in table.findall("./table/tbody/tr"):
        values = [float("".join(td.itertext()).strip()) for td in tr.findall("./td")]
        if len(values) != 7:
            raise ValueError("source DOE table changed")
        rows.append(dict(zip(names, values)))
    if len(rows) != 30 or {r["run"] for r in rows} != set(range(1, 31)):
        raise ValueError("all 30 original experimental runs required")
    methods = []
    for section in root.findall(".//sec"):
        title = (
            "".join(section.find("title").itertext())
            if section.find("title") is not None
            else ""
        )
        if title.startswith("2."):
            methods.append({"title": title, "text": " ".join(section.itertext())})
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "source.xml").write_bytes(raw)
    document = {
        "schema": "source_scoped_lotion_process_observations/v76",
        "source_url": url,
        "source_pmcid": "PMC8800138",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "doi": root.find('.//article-id[@pub-id-type="doi"]').text,
        "source_license_xml": license_xml,
        "source_table_id": "t0015",
        "rows": rows,
        "repeated_center_conditions_are_not_independent_conditions": True,
        "quantity": "reported_experimental_droplet_diameter_not_fragrance_release_or_sensory_similarity",
        "scope": "source_specific_rotor_stator_continuous_skincare_nanoemulsion",
        "methods": methods,
        "automatic_generic_lotion_transfer": False,
    }
    document["runtime_training_eligible"] = False
    document["disposition"] = (
        "research_reference_only_source_license_not_approved_for_general_runtime_training"
    )
    (a.output / "observations.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {k: v for k, v in document.items() if k not in ("rows", "methods")},
            ensure_ascii=False,
        )
    )
    for method in methods:
        print(method["text"])


if __name__ == "__main__":
    main()
