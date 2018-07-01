# recsys_2018-spotify
ACM RecSys Challenge 2018 - One Million Playlist

Introduction:
The set of programs are used within the ACM RecSys Challenge 2018 Competition. The challenge focused on music recommendations containing 1 million playlists within the online music streaming service, Spotify. The challenge involves recommending 500 song tracks for 10000 playlists. The set of programs produces recommendations based on the co-occurrence of tracks seen within other playlists.

Requirements:

	- Python Environment = 3.7
	- Packages:
		- pyspark
		- numpy
		- scipy

Files:

	1) util_track_lookup_dictionary.py
		- Description: 
			- Generates a lookup table mapping track uri to index value based on track popularity (most popular song in dataset being 0)
		- Input:
			- 1M Dataset
		- Output:
			- json file mapping track uri to index
	2) util_prepare_submission.py
		- Description: 
			- Preprocess challenge_set to convert track uri to index values
		- Input:
			- challenge_set.json
			- Lookup dictionary from program 1)
		- Output:
			- csv file
	3) util_correlation.py
		- Description:
			- Calculate co-occurrence of tracks, producing a co-occurrence matrix 
		- Input:
			- 1M Dataset
		- Output:
			- numpy array
	4) util_predictor.py
		- Description:
			- Generates recommendation based on co-occurrence matrix.
		- Input:
			- File generated from program 3 and 4
		- Output:
			- csv file containing 500 recommendations per playlist
	5) util_prepare_submission.py
		- Description:
			- Prepare submission file based on generated recommended songs from program 4)
		- Input:
			- File generated from program 1) and 4)
		- Output:
			- csv file containing recommendations ready for submission.

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
