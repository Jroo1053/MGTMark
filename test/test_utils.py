import math
import random
import unittest
import secrets
from src.lib.utils import smart_truncate, shannon_entropy
import subprocess


def gen_rand_string(word_max, word_min, num_sentences):
    res_string = ""
    for x in range(num_sentences):
        if x > 0:
            res_string += "."
        if random.random() >= .75:
            res_string += "\n"
        word_count_max = 8 + random.randrange(0, 16)
        for y in range(word_count_max):
            res_string += secrets.token_urlsafe(
                random.randrange(word_min, word_max)
            )
            res_string += " "

    return res_string


class TestUtils(unittest.TestCase):

    def test_smart_truncate(self):
        test_string = gen_rand_string(16, 8, 512)
        test_chunks = smart_truncate(
            test_string, 512, 8
        )
        assert len(test_chunks) <= 8
        chunk_lens = [len(x) for x in test_chunks]
        print(chunk_lens)

    def test_smart_truncate_small(self):
        test_string = gen_rand_string(16,8,2)
        test_chunks = smart_truncate(
            test_string,512,8
        )
        assert test_chunks

    def test_shannon_entropy(self):
        test_string = gen_rand_string(16, 8, 512)
        ent_test_file_path = "/tmp/ent_test.txt"
        with open(ent_test_file_path, "w") as ent_test:
            ent_test.write(test_string)
        ent_test_out = subprocess.run(["ent", ent_test_file_path, "-t"],
                                      capture_output=True)
        ent_test_result = float(str(ent_test_out.stdout).split(",")[8])
        test_string_entropy = shannon_entropy(test_string)
        assert math.isclose(ent_test_result, test_string_entropy, rel_tol=1e-6)


if __name__ == '__main__':
    unittest.main()
