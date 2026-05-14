"""Generate intersections.csv by looking for Avenue/Street intersections in an OSM dump."""

import csv
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Sequence

from haversine import haversine
from osm import OsmElement, OsmNode
from tqdm import tqdm


def load_osm_data(osm_json: str) -> list[OsmElement]:
    osm_data = json.load(open(osm_json))
    els = osm_data["elements"]
    return els


def get_intersection_center(nodes: Sequence[OsmNode]) -> tuple[float, float]:
    # If there are multiple nodes, it's likely they're the sides or corners of the
    # intersection. It's fine to average them, after a sanity check.
    for a, b in itertools.combinations(nodes, 2):
        d = haversine((a["lat"], a["lon"]), (b["lat"], b["lon"])) * 1000
        # Riverside Drive / 110: 42452276 <-> 42441926 111m
        if d > 120:
            # sys.stderr.write(f"  {a['id']} <-> {b['id']} {d:.0f}m\n")
            raise ValueError("Ambiguous intersection")

    num = len(nodes)
    return (
        sum(n["lat"] for n in nodes) / num,
        sum(n["lon"] for n in nodes) / num,
    )


def expand_abbrevs(s: str) -> str:
    """Expand "Ave" -> "Avenue", "St" -> "Street", etc."""
    s = re.sub(r"^St\.? ", "Saint ", s)
    s = re.sub(r"\bSt\.?(?= |$)", "Street", s)
    s = re.sub(r"\bAve\.?(?= |$)", "Avenue", s)
    s = re.sub(r"\bPl\.?(?= |$)", "Place", s)
    s = re.sub(r"\b(?<!^)Dr\.?(?= |$)", "Drive", s)  # Don't match "Dr. MLK"
    s = re.sub(r"\bRd\.?(?= |$)", "Road", s)
    s = re.sub(r"\bLn\.?(?= |$)", "Lane", s)
    s = re.sub(r"\bBlvd\.?(?= |$)", "Boulevard", s)
    return s


def main():
    input_osm, output_csv = sys.argv[1:]

    els = load_osm_data(input_osm)
    ways = [
        el
        for el in els
        if el["type"] == "way"
        and el["tags"].get("name")
        and el["tags"].get("highway") != "footway"
    ]

    id_to_way = {el["id"]: el for el in ways}
    all_nodes = [el for el in els if el["type"] == "node"]
    id_to_node = {el["id"]: el for el in all_nodes}

    node_counts = Counter[int]()
    for way in ways:
        for node in set(way["nodes"]):
            node_counts[node] += 1
    intersection_nodes = [n for n, v in node_counts.items() if v >= 2]

    node_to_ways = defaultdict(list)
    int_set = set(intersection_nodes)
    for way in ways:
        for node in way["nodes"]:
            if node in int_set:
                node_to_ways[node].append(way["id"])

    # Exclude self-intersections, e.g. W 106th St. is split into multiple ways.
    intersection_nodes = [
        n
        for n in intersection_nodes
        if len(set(id_to_way[w]["tags"]["name"] for w in node_to_ways[n])) >= 2
    ]

    way_pairs = defaultdict[tuple[str, str], set[int]](set)
    for node in intersection_nodes:
        ways = node_to_ways[node]
        for a, b in itertools.combinations(ways, 2):
            if a == b:
                continue  # can be a self-intersection, e.g. a loop
            wa = id_to_way[a]
            wb = id_to_way[b]
            name_a = wa["tags"]["name"]
            name_b = wb["tags"]["name"]
            if name_a == name_b:
                continue
            name_a = expand_abbrevs(name_a)
            name_b = expand_abbrevs(name_b)
            pair = (name_a, name_b) if name_a < name_b else (name_b, name_a)
            way_pairs[pair].add(node)

    claimed_nodes = set[int]()

    with open(output_csv, "w") as f:
        out = csv.writer(f)
        out.writerow(["Street1", "Street2", "Lat", "Lon", "Nodes"])
        for (str1, str2), intersect_node_ids in tqdm(sorted(way_pairs.items())):
            if all(n in claimed_nodes for n in intersect_node_ids):
                continue
            try:
                intersect_nodes = [id_to_node[n] for n in intersect_node_ids]
            except KeyError:
                print("Missing intersection node", str1, str2)
                raise
            try:
                lat, lng = get_intersection_center(intersect_nodes)
            except ValueError:
                # print(f"Ambiguous intersection: {str1} / {str2}: ({intersect_node_ids})")
                continue
            out.writerow(
                [
                    str1,
                    str2,
                    str(round(lat, 6)),
                    str(round(lng, 6)),
                    "/".join(str(n) for n in intersect_node_ids),
                ]
            )


if __name__ == "__main__":
    main()
