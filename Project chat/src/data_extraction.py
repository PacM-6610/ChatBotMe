import glob
import re
import csv
import os


def extract_data_whatsapp_to_csv(path_data, verbose=False):
    print(f"Extracting '{path_data}'")

    #   Compute name csv file
    list_subpart_path = path_data.split(os.sep)
    name_interlocutor = list_subpart_path[-1]
    path_csv = ""
    for subpart_path in list_subpart_path[:-2]:
        path_csv = os.path.join(path_csv, subpart_path)
    path_csv = os.path.join(path_csv, f"csv_datas")
    if not os.path.exists(path_csv):
        os.makedirs(path_csv)
    path_csv = os.path.join(path_csv, f"{name_interlocutor[:-4]}.csv")
    if verbose:
        print(path_csv)

    #   Open Csv file
    with open(path_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "message"])  # write the header row

        # Open file
        with open(path_data, "r") as file:
            lines = file.readlines()

            #   Process line by line
            for line in lines[1:]:
                #   Apply REGEX
                match = re.search(r"\[\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2}\] ([\w\s❤️]+): (.*)", line)
                #   Save match
                if match:
                    name = match.group(1)
                    message = match.group(2)

                    #   Filter message
                    if "omitted" in line:  # image, GIF, video, contact, audio
                        continue
                    elif "Missed voice call" in line:
                        continue
                    elif "This message was deleted" in line:
                        continue
                    elif "Location" in line:
                        continue
                    elif "Contact card omitted" in line:
                        continue

                    #   Save result
                    writer.writerow([remove_non_encoding_emojis(name), remove_non_encoding_emojis(message)])

                    if verbose:
                        print(f"name: {name} message: {message}")
                else:
                    if verbose:
                        print("Line does not match the pattern")


def extract_data_all_files_to_csv(path_file):
    for i, path_data in enumerate(glob.glob(f'{path_file}/*.txt')):
        extract_data_whatsapp_to_csv(path_data, verbose=False)


def remove_non_encoding_emojis(text, encoding='utf-8'):
    # Define the regular expression pattern for matching emojis
    emoji_pattern = re.compile("["
                               u"\U0001f600-\U0001f64f"  # emoticons
                               u"\U0001f300-\U0001f5ff"  # symbols & pictographs
                               u"\U0001f680-\U0001f6ff"  # transport & map symbols
                               u"\U0001f1e0-\U0001f1ff"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    # Remove all emojis that are not part of the specified encoding
    try:
        text.encode(encoding)
    except UnicodeEncodeError as e:
        text = emoji_pattern.sub(r'', text)
    return text


if __name__ == "__main__":
    extract_data_all_files_to_csv('data/whatsapp')
