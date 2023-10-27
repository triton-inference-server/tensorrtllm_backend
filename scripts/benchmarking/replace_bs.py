import argparse
import os


# Function to perform in-place substitution
def replace_line(file_path, search_string, replace_string):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if search_string in line:
                file.write("max_batch_size: {}\n".format(replace_string))
            else:
                file.write(line)


# Function to search for and replace lines in files
def replace_in_files(directory, search_string, replace_string):
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'config.pbtxt':
                file_path = os.path.join(root, file)
                replace_line(file_path, search_string, replace_string)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively replace lines in config.pbtxt files.")
    parser.add_argument("directory",
                        type=str,
                        help="The directory to search for config.pbtxt files.")
    #parser.add_argument("search_string", help="The string to search for in the lines.")
    parser.add_argument("bs_replace_value",
                        type=str,
                        help="The string to replace matching lines with.")

    args = parser.parse_args()
    search_string = "max_batch_size"
    replace_in_files(args.directory, search_string, args.bs_replace_value)


if __name__ == "__main__":
    main()
