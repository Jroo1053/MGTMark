#!python
#cython: langauge_level=3
"""

MGTMark - Machine Generated Text Detection & Obfuscation Benchmarking Tool.

Copyright (C) 2024 Elyse Frary

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
Attack Functions
"""

import random
import re
import nltk
from nltk.downloader import Downloader

nltk_down = Downloader()
if not nltk_down.is_installed("punkt"):
    nltk_down.download("punkt")

import pyximport
from libc.stdlib cimport rand, RAND_MAX

pyximport.install()


cpdef paragraph_pyx(str text, float chance):
    return _paragraph_pyx(text,chance)

cdef str _paragraph_pyx(str text, float chance):
    cdef int x = 0
    sentences = nltk.sent_tokenize(
        text
    )
    for x in range(len(sentences)):
        if (rand() / (RAND_MAX + 1.0)) <= chance:
            sentences[x] = "\n\n" + sentences[x]
    return " ".join(sentences)

cpdef whitespace_pyx(str text,float chance):
    return _whitespace_pyx(text,chance)

cdef str _whitespace_pyx(str text, float chance):
    cdef list[str] new_string =[]
    cdef int x = 0
    cdef list[str] words = text.split()
    for x in range(len(words)):
        new_string.append(words[x])
        if (rand() / (RAND_MAX + 1.0)) <= chance:
            new_string.append(" " * 2)
        else:
            new_string.append(" ")
    return "".join(new_string)

cpdef alter_numbers_pyx(str text, float chance):
    return _alter_numbers_pyx(text,chance)

cdef str _alter_numbers_pyx(str text,float chance):
    cdef list[str] new_string = []
    cdef int x = 0
    ##cdef list[str] words = [m.span() for m in re.finditer("\d+\.?\d*", text)]
    cdef list[str] words = text.split()

    for x in range(len(words)):
        if words[x].isdigit() and (rand() / (RAND_MAX + 1.0)) <= chance:
            new_string.append("".join(
                [str(random.randint(0,9)) for _ in range(len(str(words[x])))]
            ))
        else:
            new_string.append(words[x])
    return " ".join(new_string)


cdef list[int] _get_char_indexes(str text, str chr):
    cdef list[int] char_indexes = []
    cdef int x, total_chrs = text.count(chr)

    # Precompute the total number of characters
    char_indexes = [x for x in range(len(text)) if text[x] == chr]

    return char_indexes

cdef str _strat_space_pyx(str text, float chance):
    """
    Obfuscate text by strategically inserting spaces after commas.
    :param text: text to change
    :param chance: chance of attack occurring on every index of ','
    :return: new text, chars added.
    """
    if "," not in text:
        return text

    cdef list new_text = list(text)
    cdef list[int] commas = _get_char_indexes(text, ",")
    cdef int chars_swapped = 0

    for i in range(len(commas)):
        if (rand() / (RAND_MAX + 1.0)) <= chance:
            new_text.insert(commas[i] + chars_swapped + 1, " ")
            chars_swapped += 1

    return "".join(new_text)

cpdef strat_space_pyx(entry, chance):
    return _strat_space_pyx(entry,chance)


cpdef misspell_pyx(str text, dict pairs, float chance, int min_len):
    return _misspell_pyx(text, pairs, chance, min_len)


cdef tuple[str, list] _misspell_pyx(str text, dict pairs, float chance,
                                    int min_len):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in list(pairs.keys())) + r')\b'

    def replace_match(match):
        return random.choice(pairs[match.group(0)])

    replaced_text = re.sub(pattern,replace_match,text)
    return replaced_text,[]


cdef tuple[str, list] _zwsp_padding_pyx(str text, double padding_multiplier,
                                        list zwsp_chars =  ["\u200C"]):
    """
    Insert a number of Zero Width Spaces into the given string.
    Insert Len(string) * padding_multiplier characters.
    :param text: text to modify.
    :param padding_multiplier: number of ZWSP chars to add
    relative to the size of the base text.
    :param zwsp_chars: List of ZWSP chars to use.
    :return: new text number of chars changed
    """
    cdef list new_text
    cdef int i, text_len, zwsp_len

    text_len = len(text)
    zwsp_len = int(text_len * padding_multiplier)
    new_text = [*text]
    attack_indexes = []

    for i in range(zwsp_len):
        swp_index = random.randint(0, len(new_text))
        new_text.insert(
            swp_index, random.choice(zwsp_chars)
        )
        attack_indexes.append(swp_index)
    return "".join(new_text), attack_indexes


cpdef tuple[str, list] zwsp_padding_pyx(str text, double padding_multiplier,
                                        list zwsp_chars = ["\u200C"]):
    return _zwsp_padding_pyx(
        text, padding_multiplier, zwsp_chars
    )


cpdef tuple[str, list] glyph_attack_pyx(str text, dict pairs, float chance):
    return _glyph_attack_pyx(text, pairs, chance)


cdef tuple[str, list] _glyph_attack_pyx(str text, dict pairs,
                                        float glyph_chance):
    """
    Run homoglyph against a text, given a list of homoglyphs.
    :param glyph_chance: chance of inserting each glyph as a percentage.
    :param text: text to modify.
    :param pairs: list of glyphs and pairs.
    :return: modified text.
    """
    """
    This stupid loop is apparently way faster than other options for some
    reason. 
    """
    cdef list new_string = []
    cdef list modded_indexes = []
    cdef int x, tex_len = len(text)
    cdef str char, swap
    cdef list alternatives

    for x in range(tex_len):
        char = text[x]
        alternatives = pairs.get(char)
        if alternatives and (rand() / (RAND_MAX + 1.0)) <= glyph_chance:
            swap = random.choice(alternatives)
            new_string.append(swap)
            modded_indexes.append(x)
        else:
            new_string.append(char)

    return ''.join(new_string), modded_indexes
