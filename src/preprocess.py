import argparse
import bz2
import csv
import gzip
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Tuple, Union

import numpy as np

from stopwords import load_stopwords

"""
Preprocess.py prepares a corpus for training.
    - Tokenizes every word in the corpus (from local text/.jsonl files, or a Hugging Face dataset)
    - Creates a frequency count for each token
    - Indexes all words of a frequency above MIN_FREQ and creates word_to_id and id_to_word for easy index/word translation
    - Outputs corpus_index.bin, containing the entire corpus represented as integer indices
"""
MIN_FREQ = 40   # The minimum occurrence amount required in order to be considered a part of the vocabulary

def id_dtype(vocab_size: int):
    # Preprocessing writes ids with this type and training must read them back with the same one
    return np.uint16 if vocab_size <= 65535 else np.uint32

def tokenize(text: str) -> Iterator[str]:
    # Use regex to find all words (alphanumeric sequences) in the text
    word_regex = re.compile(r'\w+')
    for match in word_regex.finditer(text):
        yield match.group().lower()

# Compressed text/jsonl/csv is opened through gzip/bz2 instead of the builtin open, but otherwise
# read exactly the same way -- both support the same mode/encoding/errors/newline arguments.
_COMPRESSION_OPENERS = {'.gz': gzip.open, '.bz2': bz2.open}

# Method to stream a plain text file word-by-word, yielding cleaned tokens as it goes
def token_streamer(file_path: Path, opener=open) -> Iterator[str]:
    with opener(file_path, 'rt', encoding = 'utf-8', errors='ignore') as file:
        for line in file:
            yield from tokenize(line)

# Method to stream a .jsonl file, pulling text_key out of each record before tokenizing it
def jsonl_reader(file_path: Path, text_key: str = 'text', opener=open) -> Iterator[str]:
    with opener(file_path, 'rt', encoding = 'utf-8', errors='ignore') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole corpus
                continue
            # .get(text_key, '') would only catch a missing key — a JSON `null` value deserializes
            # to None, which .get() returns as-is since the key IS present, crashing tokenize().
            yield from tokenize(data.get(text_key) or '')

# Method to stream a .csv/.tsv file, pulling text_key out of each row before tokenizing it.
# Assumes the first row is a header, same as csv.DictReader.
def csv_reader(file_path: Path, text_key: str = 'text', opener=open, delimiter: str = ',') -> Iterator[str]:
    # newline='' is the csv module's own recommendation, so it can handle embedded newlines
    # inside quoted fields itself rather than the file iterator splitting on them first.
    with opener(file_path, 'rt', encoding='utf-8', errors='ignore', newline='') as file:
        for row in csv.DictReader(file, delimiter=delimiter):
            yield from tokenize(row.get(text_key) or '')

def parquet_reader(file_path: Path, text_key: str = 'text') -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "The 'pyarrow' package is required to read .parquet files. Install it with: pip install pyarrow"
        ) from e
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(columns=[text_key]):
        for record in batch.to_pylist():
            # Same reasoning as jsonl_reader above: columns=[text_key] means the key is always
            # present, so a null value would reach .get() as None rather than the '' default.
            yield from tokenize(record.get(text_key) or '')

# Method to stream a raw Apache Arrow IPC file (what Hugging Face `datasets` itself stores its
# on-disk cache as) -- not to be confused with .parquet, a different on-disk format.
def arrow_reader(file_path: Path, text_key: str = 'text') -> Iterator[str]:
    try:
        import pyarrow as pa
    except ImportError as e:
        raise ImportError(
            "The 'pyarrow' package is required to read .arrow files. Install it with: pip install pyarrow"
        ) from e

    with pa.memory_map(str(file_path), 'r') as source:
        # The .arrow extension doesn't say which of the two Arrow IPC sub-formats a file uses:
        # a random-access "file" with a footer (what datasets' own cache uses), or a sequential
        # "stream" without one. Try the file format first and fall back to the stream format.
        try:
            reader = pa.ipc.open_file(source)
        except pa.lib.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_stream(source)

        # RecordBatchFileReader (file format) is random-access, not an iterator, so it needs
        # get_batch(i); RecordBatchStreamReader (stream format) is already an iterator.
        batches = (reader.get_batch(i) for i in range(reader.num_record_batches)) if hasattr(reader, 'num_record_batches') else reader
        for batch in batches:
            for value in batch.column(text_key).to_pylist():
                yield from tokenize(value or '')

# Method to stream a single local file, dispatching by extension.
# Only known corpus formats are accepted; anything else raises rather than being
# silently decoded as text, which would tokenize binary garbage without erroring.
def _stream_file(file_path: Path, text_key: str) -> Iterator[str]:
    opener = open
    suffix = file_path.suffix
    if suffix in _COMPRESSION_OPENERS:
        opener = _COMPRESSION_OPENERS[suffix]
        suffix = Path(file_path.stem).suffix  # the extension underneath the compression, e.g. .jsonl in corpus.jsonl.gz

    if suffix == '.jsonl':
        yield from jsonl_reader(file_path, text_key, opener=opener)
    elif suffix == '.txt':
        yield from token_streamer(file_path, opener=opener)
    elif suffix == '.csv':
        yield from csv_reader(file_path, text_key, opener=opener, delimiter=',')
    elif suffix == '.tsv':
        yield from csv_reader(file_path, text_key, opener=opener, delimiter='\t')
    elif opener is not open:
        # .parquet/.arrow wrapped in .gz/.bz2 isn't supported: both formats already compress
        # internally, and pyarrow reads them via direct/memory-mapped file access, not a stream.
        raise ValueError(f"Compressed '{file_path.name}' isn't supported for this file type.")
    elif suffix == '.parquet':
        yield from parquet_reader(file_path, text_key)
    elif suffix == '.arrow':
        yield from arrow_reader(file_path, text_key)
    else:
        raise ValueError(
            f"Unsupported corpus file type '{file_path.suffix}' for {file_path}. "
            f"Supported extensions are .txt, .jsonl, .csv, .tsv, .parquet, .arrow "
            f"(.txt/.jsonl/.csv/.tsv may also be .gz- or .bz2-compressed)."
        )

# Method to run through every file in a local corpus directory (or single file) and yield its tokens
def _local_corpus_streamer(raw_dir: Union[str, Path], text_key: str = 'text') -> Iterator[str]:
    path = Path(raw_dir)

    if path.is_file():
        yield from _stream_file(path, text_key)
    elif path.is_dir():
        # rglob('*') recursively finds all files inside subfolders of 'path'
        for file_path in path.rglob('*'):
            # Skip hidden files (e.g. .gitkeep, .DS_Store) rather than raising on them --
            # they're not corpus content, just filesystem/git bookkeeping.
            if file_path.is_file() and not file_path.name.startswith('.'):
                yield from _stream_file(file_path, text_key)
    else:
        raise FileNotFoundError(f"Corpus input path {raw_dir} does not exist.")

# Method to stream tokens from a Hugging Face Hub dataset instead of local files.
# `datasets` is imported lazily so it stays an optional dependency.
def hf_dataset_streamer(hf_dataset: str, hf_config: Union[str, None] = None, hf_split: str = 'train', text_key: str = 'text') -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to use --hf_dataset. Install it with: pip install datasets"
        ) from e

    dataset = load_dataset(hf_dataset, name=hf_config, split=hf_split)
    for record in dataset:
        yield from tokenize(record.get(text_key, ''))

# Top-level dispatcher: streams tokens from whichever corpus source was configured.
def corpus_streamer(
    raw_dir: Union[str, Path, None] = None,
    hf_dataset: Union[str, None] = None,
    hf_config: Union[str, None] = None,
    hf_split: str = 'train',
    text_key: str = 'text',
    stopwords: frozenset = frozenset(),
) -> Iterator[str]:
    if hf_dataset is not None:
        tokens = hf_dataset_streamer(hf_dataset, hf_config, hf_split, text_key)
    elif raw_dir is not None:
        tokens = _local_corpus_streamer(raw_dir, text_key)
    else:
        raise ValueError("Either raw_dir or hf_dataset must be provided.")

    # Dropped before counting or encoding, so stopwords never reach the vocabulary or the encoded corpus
    for token in tokens:
        if token not in stopwords:
            yield token

# Method to turn a corpus of data into an indexed representation of the corpus and save the resulting word/id mappings as json files for later use.
def preprocess_corpus(
    raw_dir: Union[str, None] = "data/raw",
    processed_dir: str = "data/processed",
    hf_dataset: Union[str, None] = None,
    hf_config: Union[str, None] = None,
    hf_split: str = "train",
    text_key: str = "text",
    stopwords: str = "english",
):
    # Establish directories
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # hf_dataset, when given, takes priority over raw_dir's default so the two sources never both fire.
    source_kwargs = dict(
        raw_dir=raw_dir if hf_dataset is None else None,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        hf_split=hf_split,
        text_key=text_key,
        stopwords=load_stopwords(stopwords),
    )

    # Make a counter object to hold the streamed corpus. This allows us to cull infrequent words.
    raw_freq = Counter(corpus_streamer(**source_kwargs))

    # Build the vocab mappings from the frequency chart...
    filtered_freq: list[tuple[str, int]] = [(word, count) for word, count in raw_freq.most_common() if count >= MIN_FREQ]
    # Keep in mind that this means that the list is already sorted by frequency. Key order == id order
    vocab: Dict[str, Tuple[int, int]] = {word: (rank, count) for rank, (word, count) in enumerate(filtered_freq)}

    # ...and then store the indexes in json format.
    with open(processed_path / "sorted_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    encoding_width = 'H' if id_dtype(len(vocab)) is np.uint16 else 'I'

    # Write data to binary file
    out_file = processed_path / "corpus_index.bin"
    with open(out_file, "wb") as f_out:
        for word in corpus_streamer(**source_kwargs):
            word_id = vocab.get(word)[0] if word in vocab else None
            if word_id is not None:
                # Pack the integer into binary and store it
                f_out.write(struct.pack(encoding_width, word_id))
    print(f"Preprocessing: Corpus saved as a binary file to {out_file}.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess a raw text corpus into a vocabulary and binary token index.")

    parser.add_argument(
        "--processed_dir", type=str, default="data/processed",
        help="Directory to write sorted_vocab.json and corpus_index.bin to (default: data/processed)"
    )
    parser.add_argument(
        "--text_key", type=str, default="text",
        help="Field name containing document text, used for .jsonl files and Hugging Face datasets (default: text)"
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--raw_dir", type=str, default="data/raw",
        help="Local file or directory of raw text/.jsonl files to preprocess (default: data/raw)"
    )
    source_group.add_argument(
        "--hf_dataset", type=str, default=None,
        help="Hugging Face Hub dataset repo id to download instead of using local files, e.g. wikimedia/wikipedia"
    )

    parser.add_argument(
        "--hf_config", type=str, default=None,
        help="Dataset config/subset name required by some Hugging Face datasets, e.g. 20231101.simple"
    )
    parser.add_argument(
        "--hf_split", type=str, default="train",
        help="Dataset split to use for a Hugging Face dataset (default: train)"
    )
    parser.add_argument(
        "--stopwords", type=str, default="english",
        help="Words to drop before building the vocabulary: 'english' (built-in list), 'none', or a path to a text file "
             "with one word per line (default: english)"
    )

    args = parser.parse_args()

    preprocess_corpus(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        text_key=args.text_key,
        stopwords=args.stopwords,
    )

# Ensures code doesn't automatically run when imported into another file
if __name__ == "__main__":
    main()
