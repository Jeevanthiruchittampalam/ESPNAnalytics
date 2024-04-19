import json

f = open("players.json")
player_data = json.load(f)
f.close()
players = {}

f = open("teams.json")
team_data = json.load(f)
f.close()

superstars = [
    "C. Immobile",
    "R. Lewandowski",
    "Neymar",
    "Cristiano Ronaldo",
    "K. De Bruyne",
    "H. Kane",
    "L. Messi",
    "P. Pogba",
    "G. Bale",
    "K. Benzema",
    "Mohamed Salah",
    "A. Griezmann",
    "R. Varane",
    "E. Hazard",
    "Sergio Ramos",
    "T. Kroos",
    "P. Dybala",
]

superstar_ids = [353833, 8325, 8287, 7972, 25747, 31528]  # for manual player indexing

for i in player_data:
    short_name = i["shortName"]
    if short_name in superstars or i["wyId"] in superstar_ids:
        short_name = i["shortName"]
        club_id = i["currentTeamId"]
        country_id = i["currentNationalTeamId"]
        player_id = i["wyId"]

        players[short_name] = {
            "player_id": player_id,
            "club_id": club_id,
            "national_id": country_id,
        }

for x in players:
    print(x, players[x], end="\n\n")

# for i in team_data:
#     if i['name'] == 'Manchester City':
#         print(i['wyId'])

# for i in player_data:
#     if i['currentTeamId'] == 679:
#         player = i['wyId']
#         print(i['short_name'], player)
