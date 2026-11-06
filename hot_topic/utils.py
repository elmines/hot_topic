from typing import List, Any, Tuple, Dict, Iterable, Generator, Callable, Optional
from operator import itemgetter
import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import partial_ratio_alignment
from transformers import PreTrainedTokenizerFast

from .constants import AD_PATT

def map_key(map_func, k, dicts):
    def wrapped_map_func(d):
        return {
            k2 : map_func(v) if k2 == k else v
            for k2,v in d.items()
        }
    return map(wrapped_map_func, dicts)

_MUSIC_PATT = re.compile("|".join([
    r"\[.*?MUSIC.*?\]",
    r"\(.*?music\)",
    r"\(singing.*?\)",
]), flags=re.IGNORECASE)

WHITE_PATT = re.compile(r"\s+")

def normalize_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def drop_ad_span_lines(rows, spans, path=None):
    rows = list(rows)
    covered = set()
    line_count = len(rows)
    for span in spans:
        if (
            not isinstance(span, dict)
            or not isinstance(span.get("start"), int)
            or not isinstance(span.get("end"), int)
        ):
            if path is not None:
                print(f"{Path(path).name}: Warning: Malformed ad span: {span!r}", file=sys.stderr)
            continue
        start, end = span["start"], span["end"]
        if end < start:
            if path is not None:
                print(
                    f"{Path(path).name}: Warning: Ad span {span!r} has an invalid closed line range",
                    file=sys.stderr,
                )
            continue
        start = max(start, 0)
        end = min(end, line_count - 1)
        if start <= end:
            covered.update(range(start, end + 1))
    return [row for index, row in enumerate(rows) if index not in covered]


def align_quote(quote: str, text: str, minimum_overlap: float) -> Optional[str]:
    quote = quote.strip()
    if not quote:
        return

    alignment = partial_ratio_alignment(
        quote,
        text,
        score_cutoff=minimum_overlap,
    )
    if alignment is None:
        return None
    substr = text[alignment.dest_start:alignment.dest_end].strip()
    if not substr:
        return None
    return substr

class ContextExtractor:
    """
    Memoizes the texts of episodes,
    so we don't have to re-read the entire transcript
    off disk every time we want to extract the context
    for a quote from said transcript
    """
    def __init__(self, input_root=None, spans_root=None):
        self.__episode_texts: Dict[str, str] = dict()
        self.__input_root = Path(input_root).resolve() if input_root is not None else None
        self.__spans_root = Path(spans_root).resolve() if spans_root is not None else None
        if self.__spans_root is not None:
            if self.__input_root is None:
                raise ValueError("input_root is required when spans_root is provided")
            from .gen import load_ad_spans

            self.__load_ad_spans = load_ad_spans


    def __call__(self,
                              row: pd.Series,
                              left_context_size=4096,
                              right_context_size=4096):
        episode_path = row['episode_file']
        if episode_path not in self.__episode_texts:
            transcript_path = Path(episode_path)
            if not transcript_path.is_absolute() and self.__input_root is not None:
                transcript_path = self.__input_root / transcript_path
            with transcript_path.open(newline="") as source:
                rows = list(csv.DictReader(source))
            if self.__spans_root is not None:
                spans = self.__load_ad_spans(
                    transcript_path,
                    self.__input_root,
                    self.__spans_root,
                    len(rows),
                )
                rows = drop_ad_span_lines(rows, spans, transcript_path)
            self.__episode_texts[episode_path] = combine_rows(rows)
        text = self.__episode_texts[episode_path]
        quote_text = row['episode_quote']
        quote_start = text.index(quote_text)
        quote_end = quote_start + len(quote_text)
        left_context  = text[max(0, quote_start - left_context_size):quote_start]
        right_context = text[quote_end:quote_end + right_context_size]
        return left_context + f"<document>{quote_text}</document>" + right_context


def preprocess(rows: Iterable[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:

    # TODO: Use a joint regex for both of these?
    # Would speed things up
    # Eliminate rows with music
    rows = filter(lambda row: not _MUSIC_PATT.search(row['text']), rows)
    # Eliminate rows with ads

    # verbose version, for debugging
    def filter_func(row):
        m = AD_PATT.search(row['text'])
        if m:
            print(m.group())
            return False
        return True
    
    # The real function (overrides that earlier definition)
    filter_func = lambda row: not AD_PATT.search(row['text'])

    rows = filter(filter_func, rows)
    # Clean up large blocks of whitespace created by any earlier subs
    # Probably not needed right now
    rows = map_key(lambda text: WHITE_PATT.sub(" ", text), "text", rows)

    yield from rows

def combine_rows(rows: Iterable[Dict[str, Any]]) -> str:
    return "".join(map(itemgetter('text'), rows))

############# From Chat GPT ########################
class PartialFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def partial_format(s, **kwargs):
    return s.format_map(PartialFormatDict(kwargs))
###################################################

def extract_show_and_episode(p: os.PathLike) -> Tuple[str, str]:
    show_id = int(os.path.basename(os.path.dirname(p)))
    episode_id = os.path.basename(p).split('.')[0]
    return show_id, episode_id

def tokenized_with_trunc(tokenizer: PreTrainedTokenizerFast,
                         messages: List[Any],
                         prompt_template: str,
                         document: str,
                         max_len: int):
    base_template = messages + [{"role": "user", "content": prompt_template.format(Document="")}]
    base_prompt = tokenizer.apply_chat_template(
        base_template,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt'
    )
    base_len = base_prompt['input_ids'].shape[-1]
    # print(f"base_len = {base_len}")

    document_tokens = tokenizer.tokenize(document)
    # print(f"document_tokens = {len(document_tokens)}")
    if not document_tokens:
        return [
            tokenizer.apply_chat_template(
                messages + [{"role": "user", "content": prompt_template.format(Document=document)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
        ]

    # If the base prompt is already at or above max_len, we still need to split the
    # document into separate chunks rather than erroring out. In that case, treat the
    # document chunk size as max_len so every chunk remains a valid standalone prompt.
    remaining_capacity = max_len - base_len if base_len < max_len else max_len
    remaining_capacity = max(1, remaining_capacity)

    chunks = []
    for start in range(0, len(document_tokens), remaining_capacity):
        chunk_tokens = document_tokens[start:start + remaining_capacity]
        # assert chunk_tokens == document_tokens
        chunk_document = tokenizer.decode(tokenizer.convert_tokens_to_ids(chunk_tokens))
        chunks.append(
            tokenizer.apply_chat_template(
                messages + [{"role": "user", "content": prompt_template.format(Document=chunk_document)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
        )
        # print(f"\tchunk = {chunks[-1]['input_ids'].shape}")
    return chunks
