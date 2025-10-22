#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script CLI pour observer le serpent controle par l agent entraine actuel."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_snake_module():
    module_path = Path(__file__).with_name("snake-ia.py")
    spec = importlib.util.spec_from_file_location("snake_ia", module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Impossible de charger le module depuis {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


snake_module = load_snake_module()
DEFAULT_CHECKPOINT_PATH = snake_module.DEFAULT_CHECKPOINT_PATH
play = snake_module.play


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' n'est pas un entier valide.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("La valeur doit etre un entier strictement positif.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ouvre une fenetre PyGame pour visualiser l agent Snake entraine actuel."
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=1,
        help="Nombre de parties a visualiser (defaut: 1).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Chemin du checkpoint a charger (defaut: {DEFAULT_CHECKPOINT_PATH.name}).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Lance l'agent sans affichage (utile pour des tests rapides).",
    )
    args = parser.parse_args()

    play(
        episodes=args.episodes,
        render=not args.headless,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
