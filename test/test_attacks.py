import unittest
from src.lib.attacks import (paraphrase, glyph_attack, strat_space, misspell,
                             zwsp_padding,translate)
VALID_KEY_PATH = "../config/gpt_auth.key"

TEST_STRING = """
Wow okay, jeez. Watch out for the big guy, he knows how to write JSON.
God dam big guy over here, bet he listens to nu metal. Perchance 
"""

TEST_STRING_LARGE = """
In combat sports such as boxing, an orthodox stance is a standing position with the feet slightly wider than shoulder-width, the weight shifted towards the ball of the foot, and the hands held close to the body. The orthodox stance is considered to be one of the most effective stances in boxing. It allows a boxer to cover more ground with their footwork, and also keep their opponent at a distance by preventing them from landing clean punches.
"""

class MyTestCase(unittest.TestCase):

    def test_zwsp(self):
        zwsp_res = zwsp_padding(
            TEST_STRING, .5
        )
        res_len = len(zwsp_res[0])
        base_len = len(TEST_STRING)
        assert  res_len == base_len * 1.5

    def test_glyph(self):
        test_map = {
            "W":["w","w"]
        }
        glyph_res = glyph_attack(
            TEST_STRING,test_map,1
        )
        assert glyph_res[0][1] == "w"
        assert glyph_res[1] == 2

    def test_spacing(self):
        res = strat_space(
            TEST_STRING,1.0
        )
        assert res[0] != TEST_STRING
        assert res[1] == 3


    def test_misspell(self):
        test_map = {
            "Perchance": ["Perchancce"]
        }
        res = misspell(
            TEST_STRING, test_map, 1.0
        )
        assert res[0] != TEST_STRING
        assert "Perchancce" in res[0]
        assert  res[1] == 10


    def test_paraphrase(self):
        with open(VALID_KEY_PATH,"r") as key_file:
            key = key_file.readline()
        paraphrase_result = paraphrase(
            text=TEST_STRING,
            prompt="Rewrite these lines in the style of shakespear: ",
            model="gpt-3.5-turbo",
            auth_key=key
        )
        assert isinstance(paraphrase_result,str)
        assert len(paraphrase_result) > 0

    def test_translate(self):
        with open(VALID_KEY_PATH,"r") as key_file:
            key = key_file.readline()
        translate_result,trans_diff = translate(
            text=TEST_STRING_LARGE,
            base_lang="english",
            new_lang="chinese",
            model="gpt-3.5-turbo",
            auth_key=key
        )
        assert isinstance(translate_result,str)
        assert len(translate_result) > 0


if __name__ == '__main__':
    unittest.main()
