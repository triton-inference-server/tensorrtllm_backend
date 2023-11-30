import argparse
import csv
import os


def combine_csv_files(args):

    # List to store the data rows
    data_rows = []

    # Loop through each CSV file in the directory
    for filename in os.listdir(args.directory_path):
        if filename.endswith('.csv'):
            with open(os.path.join(args.directory_path, filename),
                      'r',
                      newline='') as csvfile:
                reader = csv.reader(csvfile)
                # Read the header (row 1) from each CSV file
                header = next(reader)
                # Read the data (row 2) from each CSV file
                data = next(reader)
                file_info = (os.path.splitext(filename)[0]).split("__")
                if file_info[2] == "v1":
                    file_info[3] = "NA"
                new_data = file_info + data
                data_rows.append(new_data)

    appended_header = [
        "Machine", "Model", "Batching Scheme", "Scheduler Policy", "Dataset",
        "REQ_RATE"
    ] + header
    # Write the combined data to the output CSV file
    with open(args.output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header fields to the output file
        writer.writerow(appended_header)
        # Write the data rows to the output file
        writer.writerows(data_rows)

    print(f"Combined data has been written to {args.output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine CSV files in a directory.")
    parser.add_argument("--directory_path",
                        type=str,
                        help="Path to the directory containing CSV files")
    parser.add_argument("--output_filename",
                        type=str,
                        help="Name of the output CSV file")

    args = parser.parse_args()

    combine_csv_files(args)


if __name__ == "__main__":
    main()
