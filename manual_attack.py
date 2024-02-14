import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from threedattack import FirstStrategy
from threedattack.script_util import (
    AttackDetails,
    add_attack_details_args,
    get_attack_details_from_parsed_args,
    run_first_strategy_attack,
)


def parse_args() -> "Args":
    parser = argparse.ArgumentParser()

    add_attack_details_args(parser)

    parser.add_argument(
        "--log-im",
        help="If set, then log the RGB images and depth maps for the best solution. Keep in mind that this somewhat slows down the attacks.",
        action="store_true",
    )

    parsed = parser.parse_args()

    attack_details = get_attack_details_from_parsed_args(parsed)
    log_im: bool = parsed.log_im
    return Args(attack_details=attack_details, log_im=log_im)


@dataclass
class Args:
    attack_details: AttackDetails
    log_im: bool


def main(args: Args) -> None:
    run_first_strategy_attack(
        attack_details=args.attack_details,
        log_im=args.log_im,
        run_repro_command=["python", "manual_attack.py"] + list(sys.argv),
    )


if __name__ == "__main__":
    main(parse_args())
