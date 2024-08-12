#!python
# cython: langauge_level=3
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
import random
from argparse import ArgumentParser

from src.lib.utils import load_config


def main():
    valid_config = load_config(args.config_file, args.samples)
    if not valid_config:
        print("Failed to load config, Exiting!")
    valid_config.load_data()
    valid_config.run_classifiers()
    valid_config.get_results()
    print()


if __name__ == "__main__":
    parser = ArgumentParser(description="MGTMark")
    parser.add_argument("-c", "--config", help="config file",
                        dest="config_file")
    parser.add_argument("-s", "--samples",
                        type=int)
    parser.add_argument("-ch", "--human-check",
                        help="Require Manual Verification",
                        dest="is_human_check",
                        default=False, action="store_true")
    args = parser.parse_args()
    main()
