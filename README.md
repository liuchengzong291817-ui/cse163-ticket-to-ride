# Ticket to Ride: Four-Player Strategy Simulator

This November 2026 research prototype compares route-planning policies in a simplified four-player game. It is an AI-assisted extension of the earlier CSE 163 team project. It does not implement the full commercial game.

## Run the demo and tests

From the repository root, with Python available:

```sh
python -B -m multiplayer.demo
python -B -m multiplayer.policy_demo
python -B -m unittest discover -s multiplayer -t .
```

The demo, simulator and tests use only the Python standard library. The archived experiments specify CPython 3.12.14 in `config/release_v1.json`.

## Repository scope

This branch publishes the simulator, strategy and experiment code, tests, configurations, and synthetic research fixtures. The November analysis report is pending a separate update. The full archived replay and byte-for-byte reproduction commands require the corresponding reference data from the original release package, which are not bundled here. The earlier course version and report remain on `main`.

