from typing import Any, Literal, TypedDict


class OsmElementBase(TypedDict):
    id: int
    tags: dict[str, Any]


class OsmNode(OsmElementBase):
    type: Literal["node"]
    lat: float
    lon: float


class OsmWay(OsmElementBase):
    type: Literal["way"]
    nodes: list[int]


class RelationMember(TypedDict):
    type: Literal["way", "node", "relation"]
    ref: int
    role: str


class OsmRelation(OsmElementBase):
    type: Literal["relation"]
    members: list[RelationMember]


OsmElement = OsmNode | OsmWay | OsmRelation
