# YAML handling — ruamel.yaml only

All YAML read/write must use `ruamel.yaml` (round-trip mode), **not**
`PyYAML`. Use `YAML(typ='rt')` for loading and dumping.
