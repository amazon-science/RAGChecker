import importlib.util
import unittest
from pathlib import Path


def load_cli_module():
    cli_path = Path(__file__).resolve().parents[1] / 'ragchecker' / 'cli.py'
    spec = importlib.util.spec_from_file_location('ragchecker_cli_under_test', cli_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CliDefaultsTest(unittest.TestCase):
    def test_bedrock_defaults_do_not_set_api_base(self):
        cli = load_cli_module()

        args = cli.get_args(
            ['--input_path', 'input.json', '--output_path', 'output.json']
        )

        self.assertEqual(args.extractor_name, cli.DEFAULT_MODEL)
        self.assertEqual(args.checker_name, cli.DEFAULT_MODEL)
        self.assertIsNone(args.extractor_api_base)
        self.assertIsNone(args.checker_api_base)
        self.assertEqual(args.metrics, [cli.DEFAULT_METRICS])

    def test_api_base_can_still_be_set_explicitly(self):
        cli = load_cli_module()

        args = cli.get_args(
            [
                '--input_path',
                'input.json',
                '--output_path',
                'output.json',
                '--extractor_api_base',
                'https://bedrock-runtime.us-east-1.amazonaws.com',
                '--checker_api_base',
                'https://example.com/v1',
            ]
        )

        self.assertEqual(
            args.extractor_api_base,
            'https://bedrock-runtime.us-east-1.amazonaws.com',
        )
        self.assertEqual(args.checker_api_base, 'https://example.com/v1')


if __name__ == '__main__':
    unittest.main()
