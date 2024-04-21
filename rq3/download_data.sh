mkdir -p data
cd data
curl -Lo players.json "https://figshare.com/ndownloader/files/15073721"
curl -Lo teams.json "https://figshare.com/ndownloader/files/15073697"
curl -Lo events.zip "https://figshare.com/ndownloader/files/14464685"
curl -Lo matches.zip "https://figshare.com/ndownloader/files/14464622"
unzip events.zip -d events
unzip matches.zip -d matches
rm events.zip
rm matches.zip
