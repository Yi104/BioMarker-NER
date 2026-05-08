from __future__ import annotations

import argparse
import csv
import gzip
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def _normalize_alias(alias: str) -> str:
    return " ".join((alias or "").strip().lower().split())


def _find_repo_root(start: Path) -> Path:
    """
    Find repository root by walking upward until project markers are found.
    This is safer than relying on a fixed parent depth.
    """
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from: {start}")


def _split_hgnc_alias_field(raw: str) -> List[str]:
    # HGNC fields may look like: "A1B|ABG|GAB"
    raw = (raw or "").strip().strip('"')
    if not raw:
        return []
    return [x.strip() for x in raw.split("|") if x.strip()]


def _write_mapping_csv(path: str, rows: Iterable[Tuple[str, str, str, str]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen: Set[Tuple[str, str, str, str]] = set()
    kept = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        kept.append(row)

    kept.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "alias", "normalized_id", "preferred_label"])
        writer.writerows(kept)
    return len(kept)


def build_gene_aliases(hgnc_path: str) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    with open(hgnc_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for rec in reader:
            hgnc_id = (rec.get("hgnc_id") or "").strip()
            symbol = (rec.get("symbol") or "").strip()
            if not hgnc_id or not symbol:
                continue
            preferred = symbol
            aliases = {symbol}
            aliases.update(_split_hgnc_alias_field(rec.get("alias_symbol", "")))
            aliases.update(_split_hgnc_alias_field(rec.get("prev_symbol", "")))
            for a in aliases:
                norm = _normalize_alias(a)
                if norm:
                    rows.append(("gene", norm, hgnc_id, preferred))
    return rows


def _is_disease_descriptor(elem: ET.Element, prefixes: Tuple[str, ...]) -> bool:
    """
    Keep only disease-related MeSH descriptors by tree-number prefix.
    Typical disease prefixes:
    - C*  : Diseases
    - F03*: Mental Disorders
    """
    for n in elem.findall("./TreeNumberList/TreeNumber"):
        t = (n.text or "").strip()
        if any(t.startswith(p) for p in prefixes):
            return True
    return False


def build_disease_aliases(mesh_desc_xml: str, disease_prefixes: Tuple[str, ...]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    for _, elem in ET.iterparse(mesh_desc_xml, events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        if not _is_disease_descriptor(elem, disease_prefixes):
            elem.clear()
            continue

        ui = (elem.findtext("./DescriptorUI") or "").strip()
        preferred = (elem.findtext("./DescriptorName/String") or "").strip()
        if not ui or not preferred:
            elem.clear()
            continue

        aliases = {preferred}
        for t in elem.findall("./ConceptList/Concept/TermList/Term/String"):
            if t.text:
                aliases.add(t.text.strip())

        mesh_id = f"MESH:{ui}"
        for a in aliases:
            norm = _normalize_alias(a)
            if norm:
                rows.append(("disease", norm, mesh_id, preferred))
        elem.clear()
    return rows


def build_chemical_aliases(compounds_tsv_gz: str, names_tsv_gz: str) -> List[Tuple[str, str, str, str]]:
    # compound_id -> (chebi_accession, preferred_name)
    compound_lookup: Dict[str, Tuple[str, str]] = {}
    with gzip.open(compounds_tsv_gz, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for rec in reader:
            cid = (rec.get("id") or "").strip()
            acc = (rec.get("chebi_accession") or "").strip()
            preferred = (rec.get("name") or "").strip()
            if cid and acc and preferred:
                compound_lookup[cid] = (acc, preferred)

    rows: List[Tuple[str, str, str, str]] = []
    # Add preferred names from compounds.tsv.gz
    for _, (acc, preferred) in compound_lookup.items():
        norm = _normalize_alias(preferred)
        if norm:
            rows.append(("chemical", norm, acc, preferred))

    # Add aliases/synonyms from names.tsv.gz
    with gzip.open(names_tsv_gz, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for rec in reader:
            cid = (rec.get("compound_id") or "").strip()
            alias = (rec.get("ascii_name") or rec.get("name") or "").strip()
            pair = compound_lookup.get(cid)
            if not pair or not alias:
                continue
            acc, preferred = pair
            norm = _normalize_alias(alias)
            if norm:
                rows.append(("chemical", norm, acc, preferred))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build normalization mapping CSVs from downloaded raw sources.")
    parser.add_argument("--hgnc", default=None)
    parser.add_argument("--mesh_desc", default=None)
    parser.add_argument("--chebi_compounds", default=None)
    parser.add_argument("--chebi_names", default=None)
    parser.add_argument("--outdir", default="data/processed/normalization")
    parser.add_argument(
        "--disease_tree_prefixes",
        default="C,F03",
        help="Comma-separated MeSH tree prefixes to keep in disease_aliases (default: C,F03).",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).parent)
    raw_dir = repo_root / "data" / "raw" / "normalization"

    # Resolve inputs with explicit arg first, then smart defaults from raw_dir.
    hgnc_path = Path(args.hgnc) if args.hgnc else raw_dir / "hgnc_complete_set.txt"
    if not hgnc_path.exists():
        candidates = sorted(raw_dir.glob("hgnc*"))
        if not candidates:
            raise FileNotFoundError(f"HGNC file not found under {raw_dir}. Pass --hgnc explicitly.")
        hgnc_path = candidates[0]

    mesh_path = Path(args.mesh_desc) if args.mesh_desc else raw_dir / "desc2026.xml"
    if not mesh_path.exists():
        candidates = sorted(raw_dir.glob("desc*.xml"))
        if not candidates:
            raise FileNotFoundError(f"MeSH descriptor XML not found under {raw_dir}. Pass --mesh_desc explicitly.")
        mesh_path = candidates[0]

    compounds_path = Path(args.chebi_compounds) if args.chebi_compounds else raw_dir / "compounds.tsv.gz"
    if not compounds_path.exists():
        candidates = sorted(raw_dir.glob("*compounds*.tsv.gz"))
        if not candidates:
            raise FileNotFoundError(f"ChEBI compounds file not found under {raw_dir}. Pass --chebi_compounds explicitly.")
        compounds_path = candidates[0]

    names_path = Path(args.chebi_names) if args.chebi_names else raw_dir / "names.tsv.gz"
    if not names_path.exists():
        candidates = sorted(raw_dir.glob("*names*.tsv.gz"))
        if not candidates:
            raise FileNotFoundError(f"ChEBI names file not found under {raw_dir}. Pass --chebi_names explicitly.")
        names_path = candidates[0]

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir

    disease_prefixes = tuple(
        p.strip() for p in args.disease_tree_prefixes.split(",") if p.strip()
    )

    gene_rows = build_gene_aliases(str(hgnc_path))
    disease_rows = build_disease_aliases(str(mesh_path), disease_prefixes)
    chem_rows = build_chemical_aliases(str(compounds_path), str(names_path))

    gene_out = os.path.join(str(outdir), "gene_aliases.csv")
    disease_out = os.path.join(str(outdir), "disease_aliases.csv")
    chem_out = os.path.join(str(outdir), "chemical_aliases.csv")

    g = _write_mapping_csv(gene_out, gene_rows)
    d = _write_mapping_csv(disease_out, disease_rows)
    c = _write_mapping_csv(chem_out, chem_rows)

    print(f"OK: gene_aliases={g} -> {gene_out}")
    print(f"OK: disease_aliases={d} -> {disease_out}")
    print(f"OK: chemical_aliases={c} -> {chem_out}")


if __name__ == "__main__":
    main()
