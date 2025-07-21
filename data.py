import json
from collections import defaultdict
import matplotlib.pyplot as plt

with open("metadata.json", "r") as file:
    dataset = json.load(file)

typecount = defaultdict(int)
language_count = defaultdict(int)
opacity_values = []
size_count = defaultdict(int)
appereance_count = defaultdict(int)
location_count = defaultdict(int)
pattern_count = defaultdict(int)
for item in dataset:
    typecount[item["content"]["type"]] += 1
    for lang in item["content"]["language"]:
        language_count[lang] += 1
    size_count[item["size"]] += 1
    appereance = item["appearance"]
    appereance_count[appereance[0]] += 1
    location_count[item["location"]] += 1
    if(item["location"] == "Repetitive"):
        pattern_count[item["pattern"]] += 1

plt.bar(typecount.keys(), typecount.values(), color='skyblue')
plt.title("Distribution of Watermark Types")
plt.ylabel("Count")
plt.show()



plt.bar(language_count.keys(), language_count.values(), color='orange')
plt.title("Languages Used in Watermarks")
plt.ylabel("Count")
plt.show()



plt.bar(size_count.keys(), size_count.values(), color='green')
plt.title("Watermark Size Distribution")
plt.ylabel("Count")
plt.show()


labels = list(appereance_count.keys())
values = list(appereance_count.values())

# Plot
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=["skyblue", "orange", "gray"])
plt.title("Watermark Appearance Types")
plt.xlabel("Appearance")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=1)
plt.tight_layout()
plt.show()

plt.bar(location_count.keys(), location_count.values(), color='black')
plt.title("Watermark Location Distribution")
plt.ylabel("Count")
plt.show()


plt.bar(pattern_count.keys(), pattern_count.values(), color='black')
plt.title("Watermark pattern distribution")
plt.ylabel("Count")
plt.show()

