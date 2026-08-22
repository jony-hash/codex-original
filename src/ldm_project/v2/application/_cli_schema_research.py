from __future__ import annotations

import argparse


def register_research(root: argparse._SubParsersAction) -> None:
    research = root.add_parser("research", help="diagnostic-only pre-registered research programs")
    research_programs = research.add_subparsers(dest="research_program", required=True)
    fiber = research_programs.add_parser("fiber-robustness", help="synthetic multi-matrix fiber diagnostics")
    fiber_actions = fiber.add_subparsers(dest="research_action", required=True)
    validate = fiber_actions.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--lp01-fixture", required=True)
    validate.add_argument("--dataset-root", required=True)
    validate.add_argument("--primary-evaluator-checkpoint", required=True)
    validate.add_argument("--secondary-evaluator-checkpoint", required=True)
    validate.add_argument("--primary-qualification-evidence", required=True)
    validate.add_argument("--secondary-qualification-evidence", required=True)
    validate.add_argument("--measurement-bundle-output", required=True)
    validate.add_argument("--measurement-evidence-output", required=True)
    validate.add_argument("--output", required=True)
    preflight = fiber_actions.add_parser("preflight")
    preflight.add_argument("--validate-evidence", required=True)
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--research-initialization-checkpoint", "--checkpoint", dest="research_initialization_checkpoint", required=True)
    preflight.add_argument("--source-training-evidence", required=True)
    preflight.add_argument("--measurement-bundle", required=True)
    preflight.add_argument("--measurement-evidence", required=True)
    preflight.add_argument("--primary-qualification-evidence", required=True)
    preflight.add_argument("--secondary-qualification-evidence", required=True)
    preflight.add_argument("--dataset-root", required=True)
    preflight.add_argument("--primary-evaluator-checkpoint", "--evaluator-checkpoint", dest="primary_evaluator_checkpoint", required=True)
    preflight.add_argument("--secondary-evaluator-checkpoint", required=True)
    preflight.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    preflight.add_argument("--output", required=True)
    screen = fiber_actions.add_parser("screen")
    screen.add_argument("--preflight-evidence", required=True)
    screen.add_argument("--config", required=True)
    screen.add_argument("--research-initialization-checkpoint", "--checkpoint", dest="research_initialization_checkpoint", required=True)
    screen.add_argument("--source-training-evidence", required=True)
    screen.add_argument("--measurement-bundle", required=True)
    screen.add_argument("--measurement-evidence", required=True)
    screen.add_argument("--primary-qualification-evidence", required=True)
    screen.add_argument("--secondary-qualification-evidence", required=True)
    screen.add_argument("--dataset-root", required=True)
    screen.add_argument("--primary-evaluator-checkpoint", "--evaluator-checkpoint", dest="primary_evaluator_checkpoint", required=True)
    screen.add_argument("--secondary-evaluator-checkpoint", required=True)
    screen.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    screen.add_argument("--output", required=True)
    select = fiber_actions.add_parser("select")
    select.add_argument("--screen-evidence", required=True)
    select.add_argument("--config", required=True)
    select.add_argument("--research-initialization-checkpoint", "--checkpoint", dest="research_initialization_checkpoint", required=True)
    select.add_argument("--source-training-evidence", required=True)
    select.add_argument("--measurement-bundle", required=True)
    select.add_argument("--measurement-evidence", required=True)
    select.add_argument("--primary-qualification-evidence", required=True)
    select.add_argument("--secondary-qualification-evidence", required=True)
    select.add_argument("--dataset-root", required=True)
    select.add_argument("--primary-evaluator-checkpoint", "--evaluator-checkpoint", dest="primary_evaluator_checkpoint", required=True)
    select.add_argument("--secondary-evaluator-checkpoint", required=True)
    select.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    select.add_argument("--output", required=True)
    closeout = fiber_actions.add_parser("closeout")
    closeout.add_argument("--selection-evidence", required=True)
    closeout.add_argument("--output", required=True)

    image = research_programs.add_parser("fiber-image", help="paired synthetic fiber image-generation diagnostics")
    image_actions = image.add_subparsers(dest="research_action", required=True)
    def image_common(command, *, validation=False):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if validation:
            command.add_argument("--lp01-fixture", required=True)
        else:
            command.add_argument("--research-initialization-checkpoint", "--checkpoint", dest="research_initialization_checkpoint", required=True)
            command.add_argument("--source-training-evidence", required=True)
            command.add_argument("--measurement-bundle", required=True)
            command.add_argument("--measurement-evidence", required=True)
    image_validate = image_actions.add_parser("validate")
    image_common(image_validate, validation=True)
    image_validate.add_argument("--measurement-bundle-output", required=True)
    image_validate.add_argument("--measurement-evidence-output", required=True)
    image_validate.add_argument("--output", required=True)
    image_baseline = image_actions.add_parser("baseline")
    image_common(image_baseline)
    image_baseline.add_argument("--validate-evidence", required=True)
    image_baseline.add_argument("--output", required=True)
    for action, upstream in (("preflight", "validate-evidence"), ("screen", "preflight-evidence")):
        command = image_actions.add_parser(action)
        image_common(command)
        command.add_argument(f"--{upstream}", dest=upstream.replace("-", "_"), required=True)
        if action == "preflight":
            command.add_argument("--baseline-evidence", required=True)
        if action in {"preflight", "screen"}:
            command.add_argument("--function-id", choices=("S0", "S1a", "S1b", "S1c"), required=True)
        if action == "screen":
            command.add_argument("--seed", type=int, choices=(5201, 5202, 5203), required=True)
            command.add_argument("--resume-checkpoint")
        command.add_argument("--output", required=True)
    image_select = image_actions.add_parser("select")
    image_common(image_select)
    image_select.add_argument("--screen-evidence", action="append", required=True)
    image_select.add_argument("--output", required=True)
    image_closeout = image_actions.add_parser("closeout")
    image_closeout.add_argument("--selection-evidence", required=True)
    image_closeout.add_argument("--output", required=True)

    attribution = research_programs.add_parser("fiber-attribution", help="no-update synthetic fiber attribution and presentation diagnostics")
    attribution_actions = attribution.add_subparsers(dest="research_action", required=True)

    def attribution_lineage(command, *, fixture=False):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--research-initialization-checkpoint", required=True)
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        if fixture:
            command.add_argument("--lp01-fixture", required=True)

    attribution_validate = attribution_actions.add_parser("validate")
    attribution_lineage(attribution_validate, fixture=True)
    attribution_validate.add_argument("--output", required=True)
    attribution_run = attribution_actions.add_parser("run")
    attribution_lineage(attribution_run)
    attribution_run.add_argument("--validation-evidence", required=True)
    attribution_run.add_argument("--output", required=True)
    attribution_report = attribution_actions.add_parser("report")
    attribution_report.add_argument("--measurement-evidence", required=True)
    attribution_report.add_argument("--v3-selection", required=True)
    attribution_report.add_argument("--digit-pipeline-variant", choices=("S0",), default="S0")
    attribution_report.add_argument("--digit-pipeline-only", action="store_true")
    attribution_report.add_argument("--output", required=True)
    attribution_closeout = attribution_actions.add_parser("closeout")
    attribution_closeout.add_argument("--measurement-evidence", required=True)
    attribution_closeout.add_argument("--presentation-evidence", required=True)
    attribution_closeout.add_argument("--output", required=True)

    interface = research_programs.add_parser(
        "fiber-interface-attribution",
        help="no-update synthetic modal-interface and optical-readout attribution",
    )
    interface_actions = interface.add_subparsers(dest="research_action", required=True)

    def interface_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--research-initialization-checkpoint", "--checkpoint", dest="research_initialization_checkpoint", required=True)
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        command.add_argument("--attribution-measurement", required=True)

    interface_validate = interface_actions.add_parser("validate")
    interface_lineage(interface_validate)
    interface_validate.add_argument("--lp01-fixture", required=True)
    interface_validate.add_argument("--output", required=True)

    interface_run = interface_actions.add_parser("run")
    interface_lineage(interface_run)
    interface_run.add_argument("--lp01-fixture", required=True)
    interface_run.add_argument("--validation-evidence", required=True)
    interface_run.add_argument("--output", required=True)

    interface_report = interface_actions.add_parser("report")
    interface_report.add_argument("--measurement-evidence", required=True)
    interface_report.add_argument("--output", required=True)

    interface_closeout = interface_actions.add_parser("closeout")
    interface_closeout.add_argument("--measurement-evidence", required=True)
    interface_closeout.add_argument("--presentation-evidence", required=True)
    interface_closeout.add_argument("--output", required=True)

    exposure = research_programs.add_parser(
        "fiber-loss-exposure-attribution",
        help="no-update synthetic fiber loss and exposure-budget attribution",
    )
    exposure_actions = exposure.add_subparsers(dest="research_action", required=True)

    def exposure_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument(
            "--research-initialization-checkpoint",
            "--checkpoint",
            dest="research_initialization_checkpoint",
            required=True,
        )
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        command.add_argument("--attribution-measurement", required=True)
        command.add_argument("--interface-validation", required=True)
        command.add_argument("--interface-measurement", required=True)
        command.add_argument("--interface-closeout", required=True)

    exposure_validate = exposure_actions.add_parser("validate")
    exposure_lineage(exposure_validate)
    exposure_validate.add_argument("--lp01-fixture", required=True)
    exposure_validate.add_argument("--output", required=True)

    exposure_run = exposure_actions.add_parser("run")
    exposure_lineage(exposure_run)
    exposure_run.add_argument("--lp01-fixture", required=True)
    exposure_run.add_argument("--validation-evidence", required=True)
    exposure_run.add_argument("--output", required=True)

    exposure_report = exposure_actions.add_parser("report")
    exposure_report.add_argument("--measurement-evidence", required=True)
    exposure_report.add_argument("--output", required=True)

    exposure_closeout = exposure_actions.add_parser("closeout")
    exposure_closeout.add_argument("--measurement-evidence", required=True)
    exposure_closeout.add_argument("--presentation-evidence", required=True)
    exposure_closeout.add_argument("--output", required=True)

    decoder = research_programs.add_parser(
        "fiber-decoder-controllability-attribution",
        help="no-update synthetic zero-phase decoder controllability attribution",
    )
    decoder_actions = decoder.add_subparsers(dest="research_action", required=True)

    def decoder_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument(
            "--research-initialization-checkpoint",
            "--checkpoint",
            dest="research_initialization_checkpoint",
            required=True,
        )
        command.add_argument("--source-training-evidence", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--v3-closeout", required=True)
        command.add_argument("--attribution-measurement", required=True)
        command.add_argument("--interface-validation", required=True)
        command.add_argument("--interface-measurement", required=True)
        command.add_argument("--interface-closeout", required=True)
        command.add_argument("--loss-exposure-validation", required=True)
        command.add_argument("--loss-exposure-measurement", required=True)
        command.add_argument("--loss-exposure-presentation", required=True)
        command.add_argument("--loss-exposure-closeout", required=True)

    decoder_validate = decoder_actions.add_parser("validate")
    decoder_lineage(decoder_validate)
    decoder_validate.add_argument("--lp01-fixture", required=True)
    decoder_validate.add_argument("--output", required=True)

    decoder_run = decoder_actions.add_parser("run")
    decoder_lineage(decoder_run)
    decoder_run.add_argument("--lp01-fixture", required=True)
    decoder_run.add_argument("--validation-evidence", required=True)
    decoder_run.add_argument("--output", required=True)

    decoder_report = decoder_actions.add_parser("report")
    decoder_report.add_argument("--measurement-evidence", required=True)
    decoder_report.add_argument("--output", required=True)

    decoder_closeout = decoder_actions.add_parser("closeout")
    decoder_closeout.add_argument("--measurement-evidence", required=True)
    decoder_closeout.add_argument("--presentation-evidence", required=True)
    decoder_closeout.add_argument("--output", required=True)

    capacity_map = research_programs.add_parser("capacity-map")
    capacity_actions = capacity_map.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "screen", "train-smoke", "confirm", "summarize", "diagnose", "qualify-shape", "closeout"):
        command = capacity_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist", "controlled_shapes", "fashion_mnist"), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cpu" if action == "closeout" else "cuda")
        if action not in {"validate", "summarize", "qualify-shape", "closeout"}:
            command.add_argument("--variant", required=True)
        else:
            command.add_argument("--variant")
        if action in {"screen", "train-smoke", "confirm", "diagnose", "qualify-shape"}:
            command.add_argument("--seed", type=int, default=7319 if action == "qualify-shape" else 1234)
        if action == "screen":
            command.add_argument("--preflight-only", action="store_true")
            command.add_argument("--stage", choices=("audit", "probe"), default="audit")
            command.add_argument("--prior-evidence")
        if action in {"train-smoke", "confirm"}:
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "diagnose":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--source-evidence", required=True)
            command.add_argument("--shape-evaluator-checkpoint")
            command.add_argument("--shape-qualification-evidence")
        if action == "summarize":
            command.add_argument("--evidence-root")
        if action == "closeout":
            command.add_argument("--comparison-config", required=True)
            command.add_argument("--evidence-root", required=True)
    o0 = research_programs.add_parser("o0-feasibility")
    o0_actions = o0.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "controls", "probe", "smoke", "confirm", "final", "summarize", "closeout"):
        command = o0_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("D0", "C0", "C1", "C2", "C3"), required=action in {"preflight", "probe", "smoke", "confirm", "final"})
        if action in {"preflight", "probe", "smoke", "confirm"}:
            command.add_argument("--seed", type=int, default=1234)
        if action in {"probe", "smoke", "confirm"}:
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "final":
            command.add_argument("--checkpoint", action="append", required=True)
            command.add_argument("--source-evidence", action="append", required=True)
        if action in {"summarize", "closeout"}:
            command.add_argument("--evidence-root", required=True)
    ident = research_programs.add_parser("o0-identifiability")
    ident_actions = ident.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "audit-update", "probe", "summarize", "closeout"):
        command = ident_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--evidence-revision", type=int, default=1)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("D0",), default="D0")
        if action in {"preflight", "audit-update", "probe"}:
            command.add_argument("--seed", type=int, default=1234)
        if action == "probe":
            command.add_argument("--prior-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action in {"summarize", "closeout"}:
            command.add_argument("--evidence-root", required=True)
    o0e = research_programs.add_parser("objective-o0")
    o0e_actions = o0e.add_subparsers(dest="research_action", required=True)
    for action in ("validate", "preflight", "calibrate", "screen", "confirm", "locked-final", "closeout"):
        command = o0e_actions.add_parser(action)
        command.add_argument("--config", required=True)
        command.add_argument("--dataset", choices=("mnist",), default="mnist")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if action != "validate":
            command.add_argument("--evaluator-checkpoint", required=True)
            command.add_argument("--primary-qualification-evidence", required=True)
            command.add_argument("--secondary-evaluator-checkpoint", required=True)
            command.add_argument("--secondary-training-evidence", required=True)
            command.add_argument("--measurement-bundle", required=True)
            command.add_argument("--measurement-evidence", required=True)
        if action in {"preflight", "calibrate", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, default=1234)
        if action == "preflight":
            command.add_argument("--validate-evidence", required=True)
            command.add_argument("--o0-i-closeout")
        if action == "calibrate":
            command.add_argument("--preflight-evidence", required=True)
        if action == "screen":
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence", required=True)
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence", required=True)
            command.add_argument("--calibration-artifact", required=True)
            command.add_argument("--calibration-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "closeout":
            command.add_argument("--source-evidence", required=True)

    wae = research_programs.add_parser(
        "electronic-o0-wae",
        help="teacher-free WAE electronic O0 lifecycle",
    )
    wae_actions = wae.add_subparsers(dest="research_action", required=True)

    def wae_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--candidate", choices=("O0-I", "O0-G"), required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)
        command.add_argument("--o0-i-closeout")

    for action in ("validate", "preflight", "screen", "confirm", "locked-final", "closeout"):
        command = wae_actions.add_parser(action)
        wae_lineage(command)
        if action in {"preflight", "screen", "confirm", "locked-final"}:
            command.add_argument("--seed", type=int, default=7301)
        if action == "preflight":
            command.add_argument("--validation-evidence", required=True)
        if action == "screen":
            command.add_argument("--validation-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "confirm":
            command.add_argument("--screen-evidence", required=True)
            command.add_argument("--resume-checkpoint")
        if action == "locked-final":
            command.add_argument("--confirm-evidence", required=True)
            command.add_argument("--checkpoint", required=True)
        if action == "closeout":
            command.add_argument("--locked-evidence", required=True)

    smf = research_programs.add_parser(
        "smf-modal-density-attribution",
        help="synthetic single-mode port-density upper-bound lifecycle",
    )
    smf_actions = smf.add_subparsers(dest="research_action", required=True)

    def smf_lineage(command):
        command.add_argument("--config", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--output", required=True)
        command.add_argument("--o0-g-decoder", required=True)
        command.add_argument("--o0-i-closeout", required=True)
        command.add_argument("--o0-g-closeout", required=True)
        command.add_argument("--fiber-closeout", required=True)
        command.add_argument("--lp01-fixture", required=True)
        command.add_argument("--primary-evaluator-checkpoint", required=True)
        command.add_argument("--secondary-evaluator-checkpoint", required=True)
        command.add_argument("--primary-qualification-evidence", required=True)
        command.add_argument("--secondary-qualification-evidence", required=True)

    smf_validate = smf_actions.add_parser("validate")
    smf_lineage(smf_validate)
    smf_validate.add_argument("--formal-config", required=True)
    smf_run = smf_actions.add_parser("run")
    smf_lineage(smf_run)
    smf_run.add_argument("--validation-evidence", required=True)
    for action in ("report", "closeout"):
        command = smf_actions.add_parser(action)
        command.add_argument("--measurement-evidence", required=True)
        if action == "closeout":
            command.add_argument("--presentation-evidence", required=True)
        command.add_argument("--output", required=True)
