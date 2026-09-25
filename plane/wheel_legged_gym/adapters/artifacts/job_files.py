"""Filesystem writes for one supervisor job; never interprets job status."""
import json
from pathlib import Path


class JobFiles:
    def __init__(self, directory):
        self.directory = Path(directory)

    def save_status(self, state):
        temporary = self.directory / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.directory / 'status.json')

    def write_text(self, name, text):
        (self.directory / name).write_text(text)

    def exists(self, path):
        return Path(path).exists()

    def read_json(self, path):
        return json.loads(Path(path).read_text())

    def write_json(self, name, value):
        self.write_text(name, json.dumps(value, indent=2) + '\n')
