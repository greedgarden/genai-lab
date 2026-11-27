import string
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / 'data' / 'names.txt'

def load_names():
    """Return the names in DATA as a list of strings, one per line."""
    with open(DATA) as f:
        names = [line.rstrip('\n') for line in f]
    return names

def count_bigrams(names):
    """Count how often each pair of adjacent characters appears in the names.

    Each name is wrapped as '.name.', so '.' marks the start and the end.
    Return a mapping from (prev_char, next_char) to its count.
    """
    bigrams = Counter()
    for name in names:
        name = f'.{name}.'
        for i in range(len(name) - 1):
            key = (name[i], name[i+1])
            bigrams[key] += 1
    return bigrams

def main():
    names = load_names()
    lengths = [len(name) for name in names]
    print(f'name lengths: {min(lengths)}~{max(lengths)}')

    bigrams = count_bigrams(names)
    for (a, b), count in bigrams.most_common(10):
        print(f'{a:>2}{b:>2}{count:>8}')
    print(f'distinct bigrams: {len(bigrams)}')

    chars = ['.'] + list(string.ascii_lowercase)
    probs = Counter()
    for a in chars:
        total_count = sum(bigrams[(a, b)] for b in chars)
        for b in chars:
            if bigrams[(a, b)] == 0:
                continue
            probs[(a, b)] = bigrams[(a, b)] / total_count

    for (a, b), f in probs.most_common(10):
        print(f'{a:>2}{b:>2}{f:>14.4g}')

    print(probs[('a', 'n')])
    row_sum = sum(probs[('a', b)] for b in chars)
    print(f'row_sum: {row_sum}')

if __name__ == "__main__":
    main()
