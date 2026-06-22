"""
Ranks SemClinBr XML files by number of distinct UMLS entity types covered.
Used to select the few-shot example files for NER experiments.

Output: ranked list with type count, annotation count, text length,
        multilabel annotation count, and the types covered.

Usage:
    python3 select_fewshot_files.py --data-dir ~/datasets/SemClinBr-xml-public-v1
    python3 select_fewshot_files.py --data-dir ~/datasets/SemClinBr-xml-public-v1 --top 10
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ENTITY_TYPES = set([
    "Abbreviation", "Acquired Abnormality", "Activity", "Age Group",
    "Amino Acid Sequence", "Amino Acid, Peptide, or Protein", "Anatomical Abnormality",
    "Antibiotic", "Bacterium", "Behavior", "Biologically Active Substance",
    "Biomedical Occupation or Discipline", "Biomedical or Dental Material",
    "Body Location or Region", "Body Part, Organ, or Organ Component",
    "Body Space or Junction", "Body Substance", "Body System", "Cell",
    "Cell or Molecular Dysfunction", "Classification", "Clinical Attribute",
    "Clinical Drug", "Congenital Abnormality", "Daily or Recreational Activity",
    "Diagnostic Procedure", "Disease or Syndrome", "Drug Delivery Device",
    "Educational Activity", "Element, Ion, or Isotope", "Enzyme", "Event",
    "Family Group", "Finding", "Fish", "Food", "Functional Concept", "Fungus",
    "Group", "Hazardous or Poisonous Substance", "Health Care Activity",
    "Health Care Related Organization", "Hormone", "Idea or Concept",
    "Immunologic Factor", "Individual Behavior", "Injury or Poisoning",
    "Inorganic Chemical", "Intellectual Product", "Laboratory Procedure",
    "Laboratory or Test Result", "Machine Activity", "Manufactured Object",
    "Medical Device", "Mental Process", "Mental or Behavioral Dysfunction",
    "Molecular Function", "Natural Phenomenon or Process", "Negation",
    "Neoplastic Process", "Nucleic Acid, Nucleoside, or Nucleotide",
    "Occupational Activity", "Organ or Tissue Function", "Organic Chemical",
    "Organism Attribute", "Organism Function", "Organization", "Pathologic Function",
    "Patient or Disabled Group", "Pharmacologic Substance", "Phenomenon or Process",
    "Physical Object", "Physiologic Function", "Plant", "Population Group",
    "Professional or Occupational Group", "Qualitative Concept", "Quantitative Concept",
    "Regulation or Law", "Research Activity", "Sign or Symptom", "Social Behavior",
    "Spatial Concept", "Substance", "Temporal Concept",
    "Therapeutic or Preventive Procedure", "Tissue", "Virus", "Vitamin",
])


def analyze_file(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    text = root.find("TEXT").text or ""

    types_found = set()
    total_ann = 0
    multilabel_ann = 0

    for ann in root.findall(".//annotation"):
        total_ann += 1
        tags = [t.strip() for t in ann.get("tag", "").split("|") if t.strip() in ENTITY_TYPES]
        types_found.update(tags)
        if len(tags) > 1:
            multilabel_ann += 1

    return {
        "file": path.name,
        "n_types": len(types_found),
        "n_annotations": total_ann,
        "n_multilabel": multilabel_ann,
        "n_chars": len(text),
        "n_words": len(text.split()),
        "types": sorted(types_found),
    }


def main():
    parser = argparse.ArgumentParser(description="Rank SemClinBr files by entity type coverage")
    parser.add_argument("--data-dir", required=True, help="Directory with SemClinBr XML files")
    parser.add_argument("--top", type=int, default=20, help="Number of top files to show (default: 20)")
    args = parser.parse_args()

    xml_files = sorted(Path(args.data_dir).glob("*.xml"))
    print(f"Analyzing {len(xml_files)} files...\n")

    results = [analyze_file(f) for f in xml_files]
    results.sort(key=lambda x: (-x["n_types"], -x["n_annotations"]))

    header = f"{'File':<12} {'Types':>6} {'Ann':>6} {'Multi':>6} {'Chars':>7} {'Words':>6}"
    print(header)
    print("-" * len(header))
    for r in results[: args.top]:
        print(
            f"{r['file']:<12} {r['n_types']:>6} {r['n_annotations']:>6} "
            f"{r['n_multilabel']:>6} {r['n_chars']:>7} {r['n_words']:>6}"
        )

    print(f"\n--- Types covered by top {min(args.top, len(results))} files ---")
    for r in results[: args.top]:
        print(f"\n{r['file']} ({r['n_types']} types):")
        print(f"  {r['types']}")


if __name__ == "__main__":
    main()
