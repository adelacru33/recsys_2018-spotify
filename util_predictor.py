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
from scipy.sparse import load_npz, dok_matrix
import numpy as np
import csv
import random
import time

RESOURCE_PATH = './resources/co-occurrence_matrix/'
MATRIX_FILE = RESOURCE_PATH + 'co-occurrence_matrix_track.npz'

SUBMISSION_PATH = './results/submission/'
INPUT_FILE = SUBMISSION_PATH + 'challenge_set.csv'
OUTPUT_FILE = SUBMISSION_PATH + 'predictions_submission_co-occurrence.csv'

VOCAB_SIZE = 2262292

TRACK_WINDOW = 5
BATCH_SIZE = 5

top_playlist_tracks = [x for x in range(1000)]

start_point = 0


def log_runtime(process, time_start):
    print('{0}! Time elapsed: {1:.2f} seconds'.format(process, time.time() - time_start))


def predict_playlist(input_tracks, track_predictor, rand_count):
    predicted_tracks = np.zeros(500, dtype=np.int64)
    unique_tracks = set(input_tracks)

    track_window = reversed(input_tracks) if len(input_tracks) <= TRACK_WINDOW else reversed(input_tracks[-TRACK_WINDOW:])
    track_count = 0

    print("Starting Prediction: Input({})".format(len(input_tracks)))
    while track_count < 500:
        itr_time = time.time()
        tracks_added = []
        for fc_track in track_window:
            row_lb = track_predictor.indptr[fc_track]
            row_ub = track_predictor.indptr[fc_track+1]
            if row_lb != row_ub:
                batch_count = 0
                while batch_count < BATCH_SIZE and track_count < 500:
                    row_data = track_predictor.data[row_lb: row_ub]
                    max_ind = np.argmax(row_data)
                    score = row_data[max_ind]
                    if score == 0:
                        break

                    track_predictor.data[row_lb + max_ind] = 0
                    track2add = track_predictor.indices[row_lb + max_ind]

                    if track2add not in unique_tracks:
                        unique_tracks.add(track2add)
                        tracks_added.append(tracks_added)
                        predicted_tracks[track_count] = track2add
                        batch_count += 1
                        track_count += 1

        if len(tracks_added) == 0:
            print("\tGenerate_Random_tracks")
            batch_count = 0
            while batch_count < BATCH_SIZE and track_count < 500:
                while top_playlist_tracks[rand_count] in unique_tracks:
                    rand_count += 1
                track2add = top_playlist_tracks[rand_count]
                unique_tracks.add(track2add)
                tracks_added.append(tracks_added)
                predicted_tracks[track_count] = track2add
                rand_count += 1
                track_count += 1
                batch_count += 1

        if track_count > 500:
            break

        track_window = input_tracks if len(tracks_added) <= TRACK_WINDOW else input_tracks[:TRACK_WINDOW]
    return predicted_tracks.tolist()


def main():
    print('Loading Data')

    s_matrix = load_npz(MATRIX_FILE)
    s_matrix = s_matrix.tocsr()

    print("Calculating Prediction Based on co-occurence")

    with open(INPUT_FILE, 'r') as csv_input:
        reader = csv.reader(csv_input)

        processed = 0
        for row in reader:
            s_time = time.time()
            pid = row[0]
            if processed >= start_point:
                print("Processing Playlist: {}".format(pid))
                tracks = row[2]

                # randomly select 5 tracks from the top most popular tracks
                rand_count = 0
                if tracks == '':
                    random.shuffle(top_playlist_tracks)
                    tracks = top_playlist_tracks[:5]
                    rand_count = 5
                else:
                    random.shuffle(top_playlist_tracks)
                    tracks = [int(x) for x in tracks.split(',')]

                pred_tracks = predict_playlist(tracks, s_matrix.copy(), rand_count)
                pred_tracks = ','.join([str(x) for x in pred_tracks])
                entry = [pid, pred_tracks]

                with open(OUTPUT_FILE, 'a') as csv_output:
                    writer = csv.writer(csv_output)
                    writer.writerow(entry)
            else:
                print("Ski[ Playlist: {}".format(pid))
            processed += 1

            log_runtime("Process Time", s_time)
    print('Done')


if __name__ == '__main__':
    main()
