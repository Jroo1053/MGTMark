from src.lib.attacks import (misspell_pyx, zwsp_padding_pyx,
                             glyph_attack_pyx,whitespace_pyx,paragraph_pyx,
                             alter_numbers_pyx,strat_space_pyx)



def paragraph_mappable(entry, args: dict, machine_label: str) -> dict:
    base_result = paragraph_pyx(
        entry[machine_label],args["chance"]
    )
    map_result = {
        "paragraph_chunks": base_result
    }
    return map_result


def whitespace_mappable(entry,args: dict, machine_label:str) -> dict:
    base_result = whitespace_pyx(
        entry[machine_label],args["chance"]
    )
    map_result = {
        "whitespace_chunks": base_result
    }
    return map_result

def alter_numbers_mappable(entry, args: dict, machine_label: str) -> dict:
    base_result = alter_numbers_pyx(
        entry[machine_label],args["chance"]
    )
    map_result = {
        "alter_numbers_chunks": base_result
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
