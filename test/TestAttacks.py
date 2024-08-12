import unittest
import pyximport

pyximport.install()
from src.lib.attacks import glyph_attack_pyx,zwsp_padding_pyx, misspell_pyx

TEST_STRING = """Wow okay, jeez. Watch out for the big guy, he knows how to write JSON.
God dam big guy over here, bet he listens to nu metal. Perchance"""





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



if __name__ == '__main__':
    unittest.main()
