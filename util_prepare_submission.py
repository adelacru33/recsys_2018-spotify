'''
Copyright 2018 Alex Dela Cruz & Kaylan Tirdad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import csv
import json


# -------------------
# Submission Param
# -------------------
TOP_TRACKS = 2000
TEAM_NAME = '{team_name}'
CHALLENGE_TASK = 'main'
TEAM_EMAIL = '{email}'
VERSION = 1

RESOURCE_PATH = './resources/lookup_tables/'
RESULTS_PATH = "./results/"
SUBMISSION_PATH = RESULTS_PATH + 'submission/'

DICTIONARY_FILE = RESOURCE_PATH + "track.index2uri.json"
INPUT_FILE = SUBMISSION_PATH + 'predictions_submission_co-occurrence.csv'
OUTPUT_FILE = SUBMISSION_PATH + 'submission-v{}.csv'.format(VERSION)


def main():
    print('Starting to pre-process challenge set')

    with open(DICTIONARY_FILE, "r") as f:
        data = f.read()
    index2uri = json.loads(data)

    with open(OUTPUT_FILE, "w") as csv_output:
        writer = csv.writer(csv_output)
        writer.writerow(['team_info', CHALLENGE_TASK, TEAM_NAME, TEAM_EMAIL])
    csv_output.close()

    with open(INPUT_FILE, 'r') as csv_input, open(OUTPUT_FILE, "a") as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)

        for row in reader:
            pid = row[0]
            print("Processing pid:{}".format(pid))

            tracks = [''] * 500
            i = 0

            for track_index in row[1].split(','):
                tracks[i] = index2uri[track_index]
                i += 1

            entry = [pid] + tracks
            writer.writerow(entry)


if __name__ == '__main__':
    main()
