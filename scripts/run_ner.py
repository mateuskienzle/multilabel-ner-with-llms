import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# SemClinBr entity types (89 UMLS semantic types)

ENTITY_TYPES = [
    "Abbreviation",
    "Acquired Abnormality",
    "Activity",
    "Age Group",
    "Amino Acid Sequence",
    "Amino Acid, Peptide, or Protein",
    "Anatomical Abnormality",
    "Antibiotic",
    "Bacterium",
    "Behavior",
    "Biologically Active Substance",
    "Biomedical Occupation or Discipline",
    "Biomedical or Dental Material",
    "Body Location or Region",
    "Body Part, Organ, or Organ Component",
    "Body Space or Junction",
    "Body Substance",
    "Body System",
    "Cell",
    "Cell or Molecular Dysfunction",
    "Classification",
    "Clinical Attribute",
    "Clinical Drug",
    "Congenital Abnormality",
    "Daily or Recreational Activity",
    "Diagnostic Procedure",
    "Disease or Syndrome",
    "Drug Delivery Device",
    "Educational Activity",
    "Element, Ion, or Isotope",
    "Enzyme",
    "Event",
    "Family Group",
    "Finding",
    "Fish",
    "Food",
    "Functional Concept",
    "Fungus",
    "Group",
    "Hazardous or Poisonous Substance",
    "Health Care Activity",
    "Health Care Related Organization",
    "Hormone",
    "Idea or Concept",
    "Immunologic Factor",
    "Individual Behavior",
    "Injury or Poisoning",
    "Inorganic Chemical",
    "Intellectual Product",
    "Laboratory Procedure",
    "Laboratory or Test Result",
    "Machine Activity",
    "Manufactured Object",
    "Medical Device",
    "Mental Process",
    "Mental or Behavioral Dysfunction",
    "Molecular Function",
    "Natural Phenomenon or Process",
    "Negation",
    "Neoplastic Process",
    "Nucleic Acid, Nucleoside, or Nucleotide",
    "Occupational Activity",
    "Organ or Tissue Function",
    "Organic Chemical",
    "Organism Attribute",
    "Organism Function",
    "Organization",
    "Pathologic Function",
    "Patient or Disabled Group",
    "Pharmacologic Substance",
    "Phenomenon or Process",
    "Physical Object",
    "Physiologic Function",
    "Plant",
    "Population Group",
    "Professional or Occupational Group",
    "Qualitative Concept",
    "Quantitative Concept",
    "Regulation or Law",
    "Research Activity",
    "Sign or Symptom",
    "Social Behavior",
    "Spatial Concept",
    "Substance",
    "Temporal Concept",
    "Therapeutic or Preventive Procedure",
    "Tissue",
    "Virus",
    "Vitamin",
]

PROMPT_ZERO_SHOT = (
    "You are an expert in clinical natural language processing, specialized in multilabel named entity recognition for healthcare texts. "
    "You must perform the task of extracting and categorizing the named entities from the clinical note provided below, considering that each named entity can be categorized in more than one class.\n\n"
    "Extract only entities that belong to the following types"
    "(use each type name exactly as written, without modification):\n{entities}\n\n"
    "Rules:\n"
    "- A span may belong to more than one type — include one annotation per (span, type) pair.\n"
    "- Only use type names from the list above. Do not invent or use any other type names.\n"
    "- If no entities are found, return an empty annotations list.\n"
    "- Return a JSON object in the format: "
    "{{\"annotations\": [{{\"label\": \"entity type\", \"text\": \"extracted span\"}}]}}\n\n"
    "IMPORTANT:\n"
    "- For every extracted span, explicitly verify whether it belongs to MORE THAN ONE label.\n"
    "- If a span matches multiple semantic types, repeat the exact same span once per label.\n"
    "- Do NOT collapse multilabel entities into a single label.\n"
)

PROMPT_FEW_SHOT = (
    "You are an expert in clinical natural language processing, specialized in multilabel named entity recognition for healthcare texts. "
    "You must perform the task of extracting and categorizing the named entities from the clinical note provided below, considering that each named entity can be categorized in more than one class.\n\n"
    "Extract only entities that belong to the following types"
    "(use each type name exactly as written, without modification):\n{entities}\n\n"
    "Rules:\n"
    "- A span may belong to more than one type — include one annotation per (span, type) pair.\n"
    "- Only use type names from the list above. Do not invent or use any other type names.\n"
    "- If no entities are found, return an empty annotations list.\n"
    "- Return a JSON object in the format: "
    "{{\"annotations\": [{{\"label\": \"entity type\", \"text\": \"extracted span\"}}]}}\n\n"
    "IMPORTANT:\n"
    "- For every extracted span, explicitly verify whether it belongs to MORE THAN ONE label.\n"
    "- If a span matches multiple semantic types, repeat the exact same span once per label.\n"
    "- Do NOT collapse multilabel entities into a single label.\n"
    "See the examples below for the expected format.\n\n"
    "{examples}"
)

# File with highest distinct entity type coverage (30/89 types) selected as the one-shot example.
# Excluded from evaluation in both zero-shot and few-shot runs to ensure metrics
# are computed on the same test set across all configurations.
DEFAULT_FEW_SHOT_FILES = ["9249.xml"]

USER_TEMPLATE = "# Clinical note:\n{input_text}"

# HAREM entity types (10 categories)

# Maps XML CATEG attribute values to Portuguese display names.
HAREM_XML_CATEG = {
    "ORGANIZACAO": "Organização",
    "PESSOA": "Pessoa",
    "LOCAL": "Local",
    "TEMPO": "Tempo",
    "ABSTRACCAO": "Abstração",
    "ACONTECIMENTO": "Acontecimento",
    "COISA": "Coisa",
    "OBRA": "Obra",
    "OUTRO": "Outro",
    "VALOR": "Valor",
}
HAREM_ENTITY_TYPES = list(HAREM_XML_CATEG.values())

HAREM_GOLD_XML_FILENAME = "CDPrimeiroHAREMMiniHAREM.xml"

# Whole document excluded from evaluation; used in full as the few-shot example.
# Selected for highest entity type coverage: 9/10 types (missing only OBRA), journalistic text.
DEFAULT_HAREM_FEW_SHOT_DOC_IDS = {"HAREM-732-05291"}

HAREM_PROMPT_ZERO_SHOT = (
    "You are an expert in natural language processing, specialized in multilabel named entity recognition for Portuguese texts. "
    "You must perform the task of extracting and categorizing the named entities from the document provided below, considering that each named entity can be categorized in more than one class.\n\n"
    "Extract only entities that belong to the following types"
    "(use each type name exactly as written, without modification):\n{entities}\n\n"
    "Rules:\n"
    "- A span may belong to more than one type — include one annotation per (span, type) pair.\n"
    "- Only use type names from the list above. Do not invent or use any other type names.\n"
    "- If no entities are found, return an empty annotations list.\n"
    "- Return a JSON object in the format: "
    "{{\"annotations\": [{{\"label\": \"entity type\", \"text\": \"extracted span\"}}]}}\n\n"
    "IMPORTANT:\n"
    "- For every extracted span, explicitly verify whether it belongs to MORE THAN ONE label.\n"
    "- If a span matches multiple semantic types, repeat the exact same span once per label.\n"
    "- Do NOT collapse multilabel entities into a single label.\n"
)
HAREM_PROMPT_FEW_SHOT = (
    "You are an expert in natural language processing, specialized in multilabel named entity recognition for Portuguese texts. "
    "You must perform the task of extracting and categorizing the named entities from the document provided below, considering that each named entity can be categorized in more than one class.\n\n"
    "Extract only entities that belong to the following types"
    "(use each type name exactly as written, without modification):\n{entities}\n\n"
    "Rules:\n"
    "- A span may belong to more than one type — include one annotation per (span, type) pair.\n"
    "- Only use type names from the list above. Do not invent or use any other type names.\n"
    "- If no entities are found, return an empty annotations list.\n"
    "- Return a JSON object in the format: "
    "{{\"annotations\": [{{\"label\": \"entity type\", \"text\": \"extracted span\"}}]}}\n\n"
    "IMPORTANT:\n"
    "- For every extracted span, explicitly verify whether it belongs to MORE THAN ONE label.\n"
    "- If a span matches multiple semantic types, repeat the exact same span once per label.\n"
    "- Do NOT collapse multilabel entities into a single label.\n"
    "See the examples below for the expected format.\n\n"
    "{examples}"
)

HAREM_USER_TEMPLATE = "# Texto:\n{input_text}"

_MSG_ZERO_SHOT = "Modo: zero-shot"

def _alt_plain_text(alt_elem):
    """Return the plain text of an ALT element (first alternative, no | separators).

    HAREM uses <ALT>word|<EM>word</EM></ALT> or <ALT><EM>span</EM>|<EM>alt</EM></ALT>
    to represent annotation ambiguity. All alternatives cover the same text span,
    so we emit only the first alternative to avoid duplicating text.
    """
    if alt_elem.text and "|" in alt_elem.text:
        return alt_elem.text.split("|")[0]
    for child in alt_elem:
        if child.tag == "EM":
            return "".join(child.itertext())
    return alt_elem.text or ""


def _extract_doc_entities(doc_elem):
    """Extract (plain_text, entities) from a gold-annotated HAREM <DOC> element.

    ALT elements contribute plain text (first alternative only) but no entity records,
    since their annotation is ambiguous by definition.
    """
    buf = ""
    entities = []

    def walk(elem):
        nonlocal buf
        if elem.text:
            buf += elem.text
        for child in elem:
            if child.tag == "ALT":
                buf += _alt_plain_text(child)
                if child.tail:
                    buf += child.tail
                continue
            if child.tag == "EM":
                start = len(buf)
                walk(child)
                end = len(buf)
                categ_str = child.get("CATEG", "")
                categ_values = [c.strip() for c in categ_str.split("|") if c.strip()]
                types = [HAREM_XML_CATEG[c] for c in categ_values if c in HAREM_XML_CATEG]
                ent_text = buf[start:end]
                if types and ent_text.strip():
                    entities.append({
                        "text": ent_text.strip(),
                        "types": types,
                        "start": start,
                        "end": end,
                    })
            else:
                walk(child)
            if child.tail:
                buf += child.tail

    walk(doc_elem)
    return buf, entities


def parse_harem_gold_xml(xml_path):
    """Parse gold HAREM XML, returning dict of doc_id → (plain_text, entities)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    docs = {}
    for doc in root.findall("DOC"):
        doc_id = doc.get("DOCID", "")
        text, entities = _extract_doc_entities(doc)
        docs[doc_id] = (text, entities)
    return docs


def split_sentences_with_offsets(text):
    """Split text into (sentence, start, end) tuples using the same regex as split_sentences."""
    result = []
    pos = 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        seg = text[pos:m.start()].strip()
        if seg:
            seg_start = text.index(seg, pos)
            result.append((seg, seg_start, seg_start + len(seg)))
        pos = m.end()
    remaining = text[pos:].strip()
    if remaining:
        seg_start = text.index(remaining, pos)
        result.append((remaining, seg_start, seg_start + len(remaining)))
    return result


def build_harem_samples(gold_docs, exclude_doc_ids=None):
    """Build sentence-level NER samples from HAREM gold docs."""
    exclude_doc_ids = exclude_doc_ids or set()
    samples = []
    for doc_id, (doc_text, gold_entities) in gold_docs.items():
        if doc_id in exclude_doc_ids:
            continue
        sentences = split_sentences_with_offsets(doc_text)
        for seg_idx, (seg_text, seg_start, seg_end) in enumerate(sentences):
            seg_entities = [
                e for e in gold_entities
                if e["start"] >= seg_start and e["end"] <= seg_end
            ]
            samples.append({
                "doc_id": doc_id,
                "segment_id": seg_idx,
                "text": seg_text,
                "gold_entities": seg_entities,
            })
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER inference with vLLM (SemClinBr or HAREM)")
    parser.add_argument("--model", required=True, help="Model name or path (HuggingFace or local)")
    parser.add_argument("--data-dir", required=True, help="Data directory (SemClinBr XML files or HAREM dataset directory)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--dataset",
        choices=["semclinbr", "harem"],
        default="semclinbr",
        help="Dataset to use (default: semclinbr)",
    )
    parser.add_argument(
        "--granularity",
        choices=["document", "sentence"],
        default="sentence",
        help="Granularity of input to the model (SemClinBr only)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model context length passed to vLLM (use to cap KV cache for models with large default contexts, e.g. Gemma 3)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files/items (for testing)")
    parser.add_argument("--few-shot", action="store_true", help="Use few-shot prompting (default: zero-shot)")
    parser.add_argument(
        "--few-shot-files",
        nargs="+",
        default=None,
        help="SemClinBr XML filenames to use as few-shot examples "
             f"(default: {DEFAULT_FEW_SHOT_FILES})",
    )
    return parser.parse_args()


def parse_xml(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    text = root.find("TEXT").text or ""

    entities = []
    for ann in root.findall(".//annotation"):
        entity_text = ann.get("text", "")
        tag_str = ann.get("tag", "")
        start = int(ann.get("start", 0))
        end = int(ann.get("end", 0))
        types = [t.strip() for t in tag_str.split("|") if t.strip()]
        entities.append({
            "text": entity_text,
            "types": types,
            "start": start,
            "end": end,
        })

    return {"doc_id": xml_path.stem, "text": text, "entities": entities}



def entities_to_annotations_json(entities: list) -> str:
    """Convert gold entity list to JSON format for use in the few-shot example.

    Each multilabel entity becomes multiple annotation objects (one per type):
    {"annotations": [{"label": ..., "text": ...}, ...]}
    """
    annotations = []
    for e in sorted(entities, key=lambda x: x.get("start", 0)):
        for t in e["types"]:
            annotations.append({"label": t, "text": e["text"]})
    return json.dumps({"annotations": annotations}, ensure_ascii=False)


def split_sentences(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def build_samples(doc: dict, granularity: str) -> list:
    if granularity == "document":
        return [{
            "doc_id": doc["doc_id"],
            "segment_id": 0,
            "text": doc["text"],
            "gold_entities": doc["entities"],
        }]

    segments = split_sentences(doc["text"])
    samples = []
    for idx, segment in enumerate(segments):
        seg_start = doc["text"].find(segment)
        seg_end = seg_start + len(segment)

        gold = [
            e for e in doc["entities"]
            if e["start"] >= seg_start and e["end"] <= seg_end
        ]

        samples.append({
            "doc_id": doc["doc_id"],
            "segment_id": idx,
            "text": segment,
            "gold_entities": gold,
        })

    return samples


def build_prompt(text: str, tokenizer, system_prompt: str, user_template: str = USER_TEMPLATE) -> str:
    user_content = user_template.format(input_text=text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


PARSE_OK = "ok"
PARSE_EMPTY = "empty"       # model returned {"annotations": []} — intentionally empty
PARSE_ERROR = "parse_error" # JSON missing or irrecoverably malformed


def parse_output(output_text: str) -> tuple:
    """Parse model output JSON: {"annotations": [{"label": ..., "text": ...}]}.

    Converts to internal format: [{"text": ..., "types": [...]}, ...]
    Multilabel entities with the same span are merged into a single types list.

    Returns (entities, status) where status is one of PARSE_OK, PARSE_EMPTY, PARSE_ERROR.
    Callers that only need entities can unpack the first element.
    """
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if not match:
        return [], PARSE_ERROR
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        raw = match.group()
        for suffix in ["]}}", "]}", "}"]:
            try:
                data = json.loads(raw + suffix)
                break
            except json.JSONDecodeError:
                continue
        else:
            return [], PARSE_ERROR

    raw_annotations = data.get("annotations", [])
    if not isinstance(raw_annotations, list):
        return [], PARSE_ERROR

    if len(raw_annotations) == 0:
        return [], PARSE_EMPTY

    merged = {}
    for ann in raw_annotations:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label", "").strip()
        text_span = ann.get("text", "").strip()
        if not label or not text_span:
            continue
        if text_span not in merged:
            merged[text_span] = []
        if label not in merged[text_span]:
            merged[text_span].append(label)

    entities = [{"text": t, "types": labels} for t, labels in merged.items()]
    return entities, PARSE_OK


def _build_harem_context(args, gold_docs, entities_str, prompt_zero_shot, prompt_few_shot):
    samples = build_harem_samples(gold_docs, exclude_doc_ids=DEFAULT_HAREM_FEW_SHOT_DOC_IDS)

    if args.few_shot:
        examples_str = ""
        for doc_id in DEFAULT_HAREM_FEW_SHOT_DOC_IDS:
            if doc_id not in gold_docs:
                continue
            doc_text, doc_entities = gold_docs[doc_id]
            response = entities_to_annotations_json(doc_entities)
            examples_str += f"# Texto example:\n{doc_text}\n\n# Saída esperada example:\n{response}\n\n"
        system_prompt = prompt_few_shot.format(entities=entities_str, examples=examples_str.strip())
        print(f"Modo: few-shot (doc IDs: {DEFAULT_HAREM_FEW_SHOT_DOC_IDS})")
    else:
        system_prompt = prompt_zero_shot.format(entities=entities_str)
        print(_MSG_ZERO_SHOT)

    if args.max_files:
        samples = samples[: args.max_files]
    print(f"Amostras HAREM para inferência: {len(samples)}")
    return samples, system_prompt


def _build_semclinbr_context(args, entities_str, prompt_zero_shot, prompt_few_shot):
    xml_files = sorted(Path(args.data_dir).glob("*.xml"))
    if args.max_files:
        xml_files = xml_files[: args.max_files]

    few_shot_names = args.few_shot_files if args.few_shot_files else DEFAULT_FEW_SHOT_FILES
    few_shot_paths = [Path(args.data_dir) / name for name in few_shot_names]
    few_shot_resolved = {p.resolve() for p in few_shot_paths}
    inference_files = [f for f in xml_files if f.resolve() not in few_shot_resolved]

    if args.few_shot:
        examples_str = ""
        for i, fsp in enumerate(few_shot_paths, 1):
            doc = parse_xml(fsp)
            response = entities_to_annotations_json(doc["entities"])
            label = f"example {i}" if len(few_shot_paths) > 1 else "example"
            examples_str += f"# Clinical note {label}:\n{doc['text']}\n\n# Expected output {label}:\n{response}\n\n"
        system_prompt = prompt_few_shot.format(entities=entities_str, examples=examples_str.strip())
        print(f"Modo: few-shot ({len(few_shot_paths)} exemplos: {[p.name for p in few_shot_paths]})")
    else:
        system_prompt = prompt_zero_shot.format(entities=entities_str)
        print(_MSG_ZERO_SHOT)

    print(f"Arquivos para inferência: {len(inference_files)}")
    print("Carregando arquivos XML...")
    docs = [parse_xml(f) for f in inference_files]
    samples = []
    for doc in docs:
        samples.extend(build_samples(doc, args.granularity))
    print(f"Total de segmentos: {len(samples)}")
    return samples, system_prompt


def main():
    args = parse_args()

    is_harem = args.dataset == "harem"

    # ── Entity types, prompts, and user template ─────────────────────────────
    if is_harem:
        entity_types = HAREM_ENTITY_TYPES
        prompt_zero_shot = HAREM_PROMPT_ZERO_SHOT
        prompt_few_shot = HAREM_PROMPT_FEW_SHOT
        user_template = HAREM_USER_TEMPLATE
    else:
        entity_types = ENTITY_TYPES
        prompt_zero_shot = PROMPT_ZERO_SHOT
        prompt_few_shot = PROMPT_FEW_SHOT
        user_template = USER_TEMPLATE

    entities_str = ", ".join(entity_types)

    # ── Load data and build system prompt ────────────────────────────────────
    if is_harem:
        gold_docs = parse_harem_gold_xml(Path(args.data_dir) / HAREM_GOLD_XML_FILENAME)
        samples, system_prompt = _build_harem_context(args, gold_docs, entities_str, prompt_zero_shot, prompt_few_shot)
    else:
        samples, system_prompt = _build_semclinbr_context(args, entities_str, prompt_zero_shot, prompt_few_shot)

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"Carregando modelo: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    prompts = [build_prompt(s["text"], tokenizer, system_prompt, user_template) for s in samples]

    print("Rodando inferência...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for sample, output in zip(samples, outputs):
        raw_output = output.outputs[0].text
        predicted, parse_status = parse_output(raw_output)
        results.append({
            "doc_id": sample["doc_id"],
            "segment_id": sample["segment_id"],
            "text": sample["text"],
            "gold": sample["gold_entities"],
            "predicted": predicted,
            "parse_status": parse_status,
            "raw_output": raw_output,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
