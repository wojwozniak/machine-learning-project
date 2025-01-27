import csv
import json

csv_file = './data/anime.csv'
json_file = './data/anime.json'

data = []
with open(csv_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append({
            "anime_id": row["anime_id"],
            "name": row["name"],
            "genre": row["genre"],
            "type": row["type"],
            "episodes": row["episodes"],
            "rating": row["rating"],
            "members": row["members"]
        })

with open(json_file, mode='w', encoding='utf-8') as jsonf:
    json.dump(data, jsonf, ensure_ascii=False, indent=4)

print(f"CSV data has been converted to JSON and saved to {json_file}.")
