"""

MGTMark - Machine Generated Text Detection & Obfuscation Benchmarking Tool.

Copyright (C) 2024 Joseph Frary

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

Mappables - Wrappers for _pyx funcs so that the can be applied to hugginface
datasets via .map(). All funcs take dict entry to obfuscate, dict of attack args
and label to access sample.
"""

from src.lib.attacks import (misspell_pyx, zwsp_padding_pyx,
                             glyph_attack_pyx, whitespace_pyx, paragraph_pyx,
                             alter_numbers_pyx, strat_space_pyx, article_delete,
                             upper_lower, comma_swap)


def upper_lower_mappable(entry, args: dict, machine_label: str) -> dict:
    base_result, indexes = upper_lower(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "upper_lower_chunks": base_result,
        "upper_lower_indexes": indexes
    }
    return map_result


def article_mappable(entry, args: dict, machine_label: str) -> dict:
    base_result, indexes = article_delete(
        entry[machine_label], args["chance"], args["articles"]
    )
    map_result = {
        "article_chunks": base_result,
        "article_indexes": indexes
    }
    return map_result


def paragraph_mappable(entry, args: dict, machine_label: str) -> dict:
    """
    Wrapper func for the paragraph attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, paragraph_indexes = paragraph_pyx(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "paragraph_chunks": base_result,
        "paragraph_indexes": paragraph_indexes
    }
    return map_result


def whitespace_mappable(entry, args: dict, machine_label: str) -> dict:
    """
    Wrapper func for the whitespace attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, whitespace_indexes = whitespace_pyx(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "whitespace_chunks": base_result,
        "whitespace_indexes": whitespace_indexes
    }
    return map_result


def alter_numbers_mappable(entry, args: dict, machine_label: str) -> dict:
    """
    Wrapper func for the alter numbers attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, indexes = alter_numbers_pyx(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "alter_number_chunks": base_result,
        "alter_number_indexes": indexes
    }
    return map_result


def misspell_mappable(entry, args: dict, machine_label: str) -> dict:
    """
    Wrapper func for the misspelling attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, indexes = misspell_pyx(
        entry[machine_label], args["pairs"], args["chance"], args["min_len"]
    )
    map_result = {
        "spelling_chunks": base_result
    }
    return map_result


def zwsp_padding_mappable(entry, args, machine_label) -> dict:
    """
    Wrapper func for the ZWSP padding attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, indexes = zwsp_padding_pyx(
        entry[machine_label], args["padding_mult"]
    )
    map_result = {
        "zwsp_chunks": base_result,
        "zwsp_indexes": indexes
    }
    return map_result


def glyph_attack_mappable(entry, args: dict, machine_label: str) -> dict:
    """
    Wrapper func for the homoglyph, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, modded_indexes = glyph_attack_pyx(entry[machine_label],
                                                   args["pairs"],
                                                   args["chance"])
    return {
        "glyph_chunks": base_result,
        "glyph_indexes": modded_indexes
    }


def strat_space_mappable(entry, args, machine_label) -> dict:
    """
    Wrapper func for the spacing attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result = strat_space_pyx(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "spacing_chunks": base_result
    }
    return map_result


def comma_swap_mappable(entry, args: dict, machine_label: str) -> dict:
    base_result, modded_indexes = comma_swap(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "comma_swap_chunks": base_result,
        "comma_swap_indexes": modded_indexes
    }
    return map_result
