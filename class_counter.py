#!/usr/bin/env python3
import csv

csv_file = "dataset/ISIC_2020_Training_GroundTruth_v2.csv"

counts = {}
with open(csv_file) as f:
    for row in csv.reader(f):
        if not row[7] in counts:
            counts[row[7]] = 0
        counts[row[7]] += 1

print(counts)