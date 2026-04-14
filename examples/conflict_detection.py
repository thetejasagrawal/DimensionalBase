"""Contradiction detection in 15 lines."""

from dimensionalbase import DimensionalBase

db = DimensionalBase()

# Subscribe to conflicts
conflicts = []
db.subscribe("**", "monitor", lambda e: conflicts.append(e) if e.type.value == "conflict" else None)

# Two agents disagree
db.put("deploy/status", "Staging is healthy and ready", owner="backend", confidence=0.9)
db.put("deploy/status", "Staging has 500 errors on /auth", owner="qa-agent", confidence=0.85)

print(f"Conflicts detected: {len(conflicts)}")
for c in conflicts:
    print(f"  {c.data['existing_entry_owner']} vs {c.data['new_entry_owner']} at {c.path}")

db.close()
