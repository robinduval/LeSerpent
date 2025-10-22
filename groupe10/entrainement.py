#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script CLI pour entrainer l agent Snake en reprenant un checkpoint existant."""

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
train = snake_module.train


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
        description="Entraine l agent Snake DQN pour N episodes (reprise automatique du checkpoint)."
    )
    parser.add_argument("episodes", type=positive_int, help="Nombre d episodes supplementaires a entrainer.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Chemin du checkpoint a utiliser (defaut: {DEFAULT_CHECKPOINT_PATH.name}).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Affiche le jeu pendant l entrainement (ralentit fortement le processus).",
    )
    parser.add_argument(
        "--render-speed",
        type=positive_int,
        help="Vitesse d affichage (FPS) lors d un entrainement rendu. Defaut: 300 ou valeur config.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore le checkpoint existant et redemarre l entrainement depuis zero.",
    )
    parser.add_argument(
        "--save-every",
        type=positive_int,
        default=1,
        help="Frequence de sauvegarde du checkpoint (en nombre d episodes).",
    )
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        render=args.render,
        render_speed=args.render_speed,
        checkpoint_path=args.checkpoint,
        resume=not args.reset,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
