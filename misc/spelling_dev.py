import random
import re
from line_profiler_pycharm import profile

TEST_STRING = """Wow okay, jeez. Watch out for the big guy, he knows how to write JSON.
God dam big guy over here, bet he listens to nu metal. Perchance"""

@profile
def misspell(text,pairs):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in list(pairs.keys())) + r')\b'

    def replace_match(match):
        return random.choice(pairs[match.group(0)])

    replaced_text = re.sub(pattern,replace_match,text)
    return replaced_text



if __name__ == "__main__":
    test_pairs = {
        "wow": ["whoow"],
        "cool": ["pool"],
        "jeez": ["jazz"]
    }
    for x in range(1000):
        misspell(TEST_STRING,test_pairs)