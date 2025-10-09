
# -*- coding: utf-8 -*-
"""
convert_signed_dataset.py
--------------------------
Usage:
    python convert_signed_dataset.py --input ./experiment-data/raw_bitcoinalpha.csv \\
                                     --output ./experiment-data/bitcoin_alpha.edgelist

Description:
    Converts signed network data with numeric weights (comma-, tab-, semicolon-,
    or space-delimited) into a simplified 3-column edgelist:
        u v s
    where s ∈ {1, -1} (zeros dropped by default).

Options:
    --u_col / --v_col / --w_col : 0-based column indices (auto-detected if omitted)
    --keep_zero                  : keep edges with weight == 0 (written with sign 0)
    --drop_self_loops            : drop u == v edges (default: ON)
    --no_dedup                   : do not drop duplicate edges (default drops duplicates)
"""

import argparse
import csv
import math
import os

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def detect_delimiter(path):
    # Try csv.Sniffer first; fall back heuristics
    with open(path, 'r', newline='') as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';',' '])
        return dialect.delimiter
    except Exception:
        for d in [',','\t',';',' ']:
            if d in sample:
                return d
    return ','

def guess_columns(row):
    # Return indices (u_col, v_col, w_col) by picking first 3 numeric columns
    num_idxs = [i for i, val in enumerate(row) if is_number(val)]
    if len(num_idxs) >= 3:
        return num_idxs[0], num_idxs[1], num_idxs[2]
    # fallback: classic 0,1,2 if present
    if len(row) >= 3:
        return 0, 1, 2
    return None

def convert_signed_edges(input_path, output_path, u_col=None, v_col=None, w_col=None,
                         keep_zero=False, drop_self_loops=True, dedup=True):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    delim = detect_delimiter(input_path)

    total = 0
    written = 0
    skipped_non_numeric = 0
    skipped_zeros = 0
    skipped_self = 0

    # deduplication
    seen = set()

    with open(input_path, 'r', newline='') as f, open(output_path, 'w', newline='') as out:
        reader = csv.reader(f, delimiter=delim)
        writer = out.write

        first_row = True
        cols = (u_col, v_col, w_col)

        for row in reader:
            if not row or all((c.strip() == '' for c in row)):
                continue

            # Attempt to auto-detect columns on the first non-empty row if not provided
            if first_row and None in cols:
                guess = guess_columns(row)
                if guess is None:
                    # skip header and continue
                    first_row = False
                    continue
                else:
                    uci, vci, wci = guess
                    cols = (uci if u_col is None else u_col,
                            vci if v_col is None else v_col,
                            wci if w_col is None else w_col)
                first_row = False

            uci, vci, wci = cols

            # Bounds check
            if uci >= len(row) or vci >= len(row) or wci >= len(row):
                # likely a header line or malformed line
                skipped_non_numeric += 1
                continue

            u_str, v_str, w_str = row[uci], row[vci], row[wci]
            if not (is_number(u_str) and is_number(v_str) and is_number(w_str)):
                skipped_non_numeric += 1
                continue

            u = int(float(u_str))
            v = int(float(v_str))
            w = float(w_str)

            total += 1

            if drop_self_loops and u == v:
                skipped_self += 1
                continue

            if w == 0.0 and not keep_zero:
                skipped_zeros += 1
                continue

            s = 0
            if w > 0:
                s = 1
            elif w < 0:
                s = -1
            else:
                s = 0  # only if keep_zero

            tup = (u, v, s)
            if dedup:
                if tup in seen:
                    continue
                seen.add(tup)

            writer(f"{u} {v} {s}\n")
            written += 1

    pos = sum(1 for (_,_,s) in seen if s == 1) if dedup else None
    neg = sum(1 for (_,_,s) in seen if s == -1) if dedup else None
    zero = sum(1 for (_,_,s) in seen if s == 0) if dedup else None

    print(f"Input: {input_path}")
    print(f"Detected delimiter: {repr(delim)}")
    print(f"Parsed rows: {total}")
    print(f"Written edges: {written}")
    if dedup:
        print(f"Unique (after dedup): pos={pos}, neg={neg}, zero={zero}")
    print(f"Skipped: non-numeric/headers={skipped_non_numeric}, zeros(dropped)={skipped_zeros}, self-loops={skipped_self}")
    print(f"Output: {output_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to raw dataset (csv/tsv/space-delimited)')
    ap.add_argument('--output', required=True, help='Path to output .edgelist (u v s)')
    ap.add_argument('--u_col', type=int, default=None, help='0-based column index for source node')
    ap.add_argument('--v_col', type=int, default=None, help='0-based column index for target node')
    ap.add_argument('--w_col', type=int, default=None, help='0-based column index for weight/sign')
    ap.add_argument('--keep_zero', action='store_true', help='Keep edges with weight == 0 (sign 0)')
    ap.add_argument('--drop_self_loops', action='store_true', default=True, help='Drop u==v edges (default on)')
    ap.add_argument('--no_dedup', action='store_true', help='Do not drop duplicate edges')
    args = ap.parse_args()

    convert_signed_edges(
        input_path=args.input,
        output_path=args.output,
        u_col=args.u_col,
        v_col=args.v_col,
        w_col=args.w_col,
        keep_zero=args.keep_zero,
        drop_self_loops=args.drop_self_loops,
        dedup=not args.no_dedup
    )
