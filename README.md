# Requirements

- Run `pip install -r requirements.txt` to install the required packages.

# Downloading the dataset

### RQ1
The dataset for RQ1 is nearly a gigabyte in size, so we cannot provide it on GitHub. Either run the provided download script, or download the data manually.

##### Run the download script
  - Navigate to the `rq1` directory
  - If on Windows, run `download_data.bat` and unzip all files
  - If on Unix, run `download_data.sh`

##### Downloading manually
- If you are unable to run the script, download the `Players`, `Teams`, `Events` and `Matches` files from [FigShare](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2).

After downloading the data, ensure your `rq1` directory looks like this:
```
rq1/
├── data/
│   ├── events/
│   │   ├── events_England.json
│   │   ├── ...
│   │   └── events_World_Cup.json
│   ├── matches/
│   │   ├── matches_England.json
│   │   ├── ...
│   │   └── matches_World_Cup.json
│   ├── players.json
│   └── teams.json
├── ...
└── rq1.ipynb
```

### RQ2 and RQ3
English Premier League datasets are provided in respective directories.

