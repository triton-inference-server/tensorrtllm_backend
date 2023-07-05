#! /usr/bin/env python3
from argparse import ArgumentParser
from string import Template


def main(file_path, substitutions):
    with open(file_path) as f:
        pbtxt = Template(f.read())

    sub_dict = {}
    for sub in substitutions.split(","):
        key, value = sub.split(":")
        sub_dict[key] = value

    pbtxt = pbtxt.safe_substitute(sub_dict)

    with open(file_path, "w") as f:
        f.write(pbtxt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file_path", help="path of the .pbtxt to modify")
    parser.add_argument(
        "substitutions",
        help=
        "substitions to perform, in the format variable_name_1:value_1,variable_name_2:value_2..."
    )
    args = parser.parse_args()

    main(**vars(args))
