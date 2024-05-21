import unittest
import secrets
from src.lib.utils import load_auth_creds
from src.lib.models import APIPipeline


VALID_KEY_PATH = "../config/gpt_auth.key"
VALID_KEY_PATH_ORIG = "../config/orig_auth.key"
VALID_KEY_PATH_WINSTON = "../config/winston_auth.key"
TEST_CONTENT_PATH = "./test_content.txt"

class TestAPIPipeline(unittest.TestCase):

    def test_load_auth_creds_gpt3_valid(self):
        assert load_auth_creds(VALID_KEY_PATH,model="GPT3")


    def test_load_auth_creds_gpt3_invalid(self):
        fake_token = secrets.token_urlsafe(16)
        with open("/tmp/test.key", "w") as test_key:
            test_key.write(fake_token)
        try:
            load_auth_creds(VALID_KEY_PATH,model="GPT3")
        except ValueError:
            assert True


    def test_winston_call_invalid(self):
        fake_token = secrets.token_urlsafe(16)
        with open("/tmp/test.key", "w") as test_key:
            test_key.write(fake_token)
        try:
            test_pipe = APIPipeline(
                provider="WINSTON",auth_creds="/tmp/test.key"
            )
        except ValueError:
            assert True
            return
        assert False

    def test_orig_call_invalid(self):
        fake_token = secrets.token_urlsafe(16)
        with open("/tmp/test.key", "w") as test_key:
            test_key.write(fake_token)
        try:
            test_pipe = APIPipeline(
                provider="ORIG",auth_creds="/tmp/test.key"
            )
        except ValueError:
            assert True
            return
        assert False

    def test_orig_call(self):
        with open(TEST_CONTENT_PATH,"r") as test_content_file:
            test_content = test_content_file.read()
        test_pipeline = APIPipeline(
            provider="ORIG",auth_creds=VALID_KEY_PATH_ORIG
        )
        detect_result = test_pipeline(
            test_content,max_length=512
        )
        assert detect_result[0]["label"] == "Real" or detect_result[0]["label"] == "Fake"
        assert isinstance(detect_result["score"],float)

    def test_winston_call(self):
        with open(TEST_CONTENT_PATH,"r") as test_content_file:
            test_content = test_content_file.read()
        test_pipeline = APIPipeline(
            provider="WINSTON",auth_creds=VALID_KEY_PATH_WINSTON
        )
        detect_result = test_pipeline(
            test_content,max_length=512
        )
        assert detect_result[0]["label"] == "Real" or detect_result[0]["label"] == "Fake"
        assert isinstance(detect_result[0]["score"],float)





if __name__ == '__main__':
    unittest.main()
