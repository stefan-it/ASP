import json
import sys
import logging

from flair.data import Sentence
from flair.datasets import NER_HIPE_2022

from pathlib import Path
from typing import List

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__file__)


def prepare_ajmc_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        lines = f_p.readlines()

    with open(file_out, "wt") as f_out:
        # Add missing newline after header
        f_out.write(lines[0] + "\n")

        for line in lines[1:]:
            if line.startswith(" \t"):
                # Workaround for empty tokens
                continue

            line = line.strip()

            # HIPE-2022 late pre-submission fix:
            # Our hmBERT model has never seen Fraktur, so we replace long s
            line = line.replace("ſ", "s")

            # Add "real" document marker
            if add_document_separator and line.startswith(document_separator):
                f_out.write("-DOCSTART- O\n\n")

            f_out.write(line + "\n")

            if eos_marker in line:
                    f_out.write("\n")

    print("Special preprocessing for AJMC has finished!")

def hipe2022_to_json(base_folder: Path):
    logger.info(f"converting hipe2022 to json in {base_folder}")

    hipe2022_datasets, hipe2022_types = {}, {}

    for name in ["dev", "test", "train"]:
        logger.info(f"processing {base_folder}/{name}.txt")
        data = open(f"{base_folder}/{name}.txt").readlines()
        
        dataset = []
        idx, start, current_type, doc = -1, None, None, None
        for line in data:
            line = line.strip()

            if line.startswith("TOKEN\tNE-COARSE-LIT") or line.startswith("# "):
                # Ignore header
                continue

            if line.startswith("-DOCSTART-"):
                if doc is not None:
                    # when extended is not the same as tokens
                    # mark where to copy from with <extra_id_22> and <extra_id_23>
                    # E.g.
                    # Extract entities such as apple, orange, lemon <extra_id_22> Give me a mango . <extra_id_23>
                    # See ace05_to_json.py for example of extending the input
                    doc["extended"] = doc["tokens"]
                    dataset.append(doc)
                doc = {
                    "tokens": [], # list of tokens for the model to copy from
                    "extended": [], # list of input tokens. Prompts, instructions, etc. go here
                    "entities": [] # list of dict:{"type": type, "start": start, "end": end}, format: [start, end)
                }
                idx, start = -1, None
                continue
            elif line == "":
                if doc and len(doc["tokens"]) > 800 and name == "train": # clip
                    if doc is not None:
                        doc["extended"] = doc["tokens"]
                        dataset.append(doc)
                    doc = {
                        "tokens": [],
                        "extended": [],
                        "entities": []
                    }
                    idx, start = -1, None
                    continue
                # new sentence
                pass
            else:
                idx += 1
                items = line.split()
                assert len(items) >= 10, line

                token = items[0]
                bio_tag = items[1]
                doc["tokens"].append(items[0])

                if bio_tag[0] == 'I':
                    pass
                elif bio_tag[0] == 'O':
                    if start is not None:
                        doc['entities'].append(
                            {
                                "type": current_type,
                                "start": start,
                                "end": idx
                            }
                        )
                    start = None
                elif bio_tag[0] == 'B':
                    if start is not None:
                        doc['entities'].append(
                            {
                                "type": current_type,
                                "start": start,
                                "end": idx
                            }
                        )
                    start = idx
                    current_type = bio_tag[2:]
                    hipe2022_types[current_type] = {
                        "short": current_type
                    }
        dataset.append(doc)
        hipe2022_datasets[name] = dataset

    for name in hipe2022_datasets:
        logger.info(f"maximum input length: {max([len(x['extended']) for x in hipe2022_datasets[name]])}")
        logger.info(f"saving {len(hipe2022_datasets[name])} documents to {base_folder}/hipe2022_{name}.json")
        with open(f"{base_folder}/hipe2022_{name}.json", 'w') as fout:
            json.dump(hipe2022_datasets[name], fout)
    
    with open(f"{base_folder}/hipe2022_types.json", 'w') as fout:
        logger.info(f"saving types to {base_folder}/hipe2022_types.json")
        json.dump({"entities": hipe2022_types}, fout)

if __name__ == "__main__":
    preproc_fn = prepare_ajmc_corpus
    corpus = NER_HIPE_2022(dataset_name="ajmc", language="en", preproc_fn=preproc_fn, add_document_separator=True)
    base_folder = Path(corpus.name)

    hipe2022_to_json(base_folder)
