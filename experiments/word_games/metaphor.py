"""Model-selected metaphor from public task structure, before private inputs."""
from dataclasses import dataclass, asdict
import json
import re

PUBLIC_TASK = """Choose a metaphor that helps two agents coordinate this task:
Distinct objects have stable identities and home locations. Objects can occupy
one numbered slot at a location until a return deadline. Locations have inherited
compatibility rules; some objects require an explicit narrow approval. Moves,
returns and condition checks preserve identity and exclusive occupancy.
The receiving agent needs only authorized facts/actions in the new metaphor.
Choose the metaphor yourself to suit these relationships.
You have no private source data. Return exactly these JSON string fields:
name, entity, location, slot, place_verb, return_verb, approval, deadline.
Use short lowercase English phrases (letters, spaces and hyphens only, maximum
40 characters). Entity/location/slot/approval/deadline must be distinct role names.
Use imperative verbs for place_verb and return_verb. No extra fields.
"""


@dataclass(frozen=True)
class Metaphor:
    name: str
    entity: str
    location: str
    slot: str
    place_verb: str
    return_verb: str
    approval: str
    deadline: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not re.fullmatch(r"[a-z][a-z -]{0,39}", v) for v in asdict(self).values()):
            raise ValueError("bounded role labels required")
        if len({self.entity, self.location, self.slot, self.approval, self.deadline}) != 5:
            raise ValueError("distinct role labels required")


def select(client, out):
    return Metaphor(**json.loads(client.complete(PUBLIC_TASK, out, max_tokens=256)))
