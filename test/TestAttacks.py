import unittest
import pyximport
import nltk
from nltk.downloader import Downloader

nltk_down = Downloader()
if not nltk_down.is_installed("punkt"):
    nltk_down.download("punkt")



pyximport.install()
from src.lib.attacks import (
    glyph_attack_pyx,zwsp_padding_pyx, misspell_pyx,
    whitespace_pyx,paragraph_pyx,alter_numbers_pyx, article_delete,
    upper_lower)

TEST_STRING = """Wow okay, jeez. Watch out for the big guy, he knows how to write JSON.
God dam big guy over here, bet he listens to nu metal.
Perchance. Yeah 1801 was a bad year idk, maybe 2002 or even 202020 that would
be far of though so we should be okay. The Cool dude jumps over a badly coded loop for An stupid reason. 
"""


class TestAttacks(unittest.TestCase):
    def test_glyph_attack(self):
        test_map = {
            "W":["w","w"]
        }
        glyph_res, glyph_indexes = glyph_attack_pyx(
            TEST_STRING,test_map,1
        )
        print(glyph_res)
        assert glyph_res[0] == "w"
        assert glyph_indexes[0] == 0

    def test_zwsp_attack(self):
        zwsp_res, zwsp_indexs = zwsp_padding_pyx(
            TEST_STRING,0.5,zwsp_chars=["\u200C"]
        )
        print(zwsp_res)
        print(len(zwsp_res))
        print(len(TEST_STRING))
        assert len(zwsp_res) >= int(len(TEST_STRING) * 1.5)

    def test_misspell(self):
        test_pairs = {
            "wow":["whoow"],
            "cool":["pool"]
        }

        mispell_res, miss_indexes = misspell_pyx(
            TEST_STRING,test_pairs,1,2
        )

    def test_whitespace(self):
        whitespace_res, whitespace_indexes = whitespace_pyx(
            TEST_STRING,.5
        )
        assert whitespace_indexes
        for x in whitespace_indexes:
            if whitespace_res[x] != " ":
                assert False

    def test_paragraph(self):
        orig_sentences = len(nltk.sent_tokenize(
            TEST_STRING
        ))

        para_res,para_indexes = paragraph_pyx(
            TEST_STRING,.5
        )
        new_newlines = para_res.count("\n")
        assert para_indexes
        for x in para_indexes:
            if para_res[x] != "\n":
                assert False

    def test_alter_numbers(self):
        number_res,number_indexes = alter_numbers_pyx(
            TEST_STRING, 1
        )
        assert number_indexes
        for index in number_indexes:
            if not number_res[index].isdigit():
                assert False

    def test_delete_article(self):
        article_res, article_indexes = article_delete(
            TEST_STRING,1
        )
        print(article_res)
        assert "the" not in article_res

    def test_upper_lower(self):
        case_res,case_indexes = upper_lower(
            TEST_STRING,1
        )
        assert case_res
        assert case_indexes
        for x in case_res:
            if x.isupper():
                assert False


if __name__ == '__main__':
    unittest.main()
