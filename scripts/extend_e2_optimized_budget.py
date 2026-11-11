"""Record a user-authorized additive budget amendment; never train or reset history."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def checked_hours(value, confirmed):
    if not confirmed:
        raise ValueError("explicit_user_budget_confirmation_required")
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive_finite_total_hours_required")
    return float(value)


def apply_extension(args, opt, mirror_factory, session_factory):
    hours = checked_hours(args.total_hours, args.confirmed_by_user)
    registration, registration_sha = opt.read_registration()
    with tempfile.TemporaryDirectory(prefix="e2-explicit-budget-amendment-") as temporary:
        output = Path(temporary) / "campaign"
        with mirror_factory(opt.REPOSITORY, opt.BRANCH, opt.BRANCH) as mirror:
            mirror.restore(output)
            config_path = output / "campaign.json"
            if not config_path.is_file():
                raise ValueError("existing_optimized_campaign_required")
            config = json.loads(config_path.read_text())
            plan = config["plan"]
            if plan["version"] != opt.VERSION or plan["registration_sha256"] != registration_sha:
                raise ValueError("existing_campaign_implementation_mismatch")
            if plan["remote_storage"] != {"repository": opt.REPOSITORY, "branch": opt.BRANCH, "prefix": opt.BRANCH}:
                raise ValueError("campaign_remote_binding_changed")
            if any(plan["source_sha256"].get(name) != sha
                   for name, sha in registration["implementation_sha256"].items()):
                raise ValueError("campaign_implementation_changed")
            initial = plan["initial_controls"]
            # Always replay with the original sealed controls. The extension is
            # an event, not a rewrite of plan/campaign.json or the legacy cache.
            with session_factory(output, plan,
                    session_budget_seconds=initial["session_budget_seconds"],
                    campaign_budget_seconds=initial["campaign_budget_seconds"],
                    max_storage_bytes=initial["storage_bytes"]) as session:
                if (session._unclosed_allocations
                        or set(session._sessions) - session._finished_sessions):
                    raise ValueError("unclosed_campaign_session_requires_reconciliation")
                legacy_spent = plan["legacy_cache"]["spent_seconds"]
                current_total = legacy_spent + session.campaign_budget_seconds
                requested_total = hours * 3600.0
                if requested_total < current_total - 1e-6:
                    raise ValueError("budget_reduction_refused")
                if abs(requested_total - current_total) <= 1e-6:
                    return {"status": "already_registered", "total_hours": current_total / 3600,
                            "remote_commit": mirror._head, "training_started": False}
                status_path = output / "status.json"
                if not status_path.is_file():
                    raise ValueError("closed_budget_pause_report_required")
                previous = json.loads(status_path.read_text())
                if (previous.get("status") not in ("paused_campaign_budget", "budget_extended_awaiting_resume")
                        or previous.get("plan_sha256") != session.plan_sha256):
                    raise ValueError("confirmed_budget_pause_required")
                reason = ("Explicit user authorization via Colab input: combined total "
                          + format(current_total / 3600, ".12g") + " -> "
                          + format(hours, ".12g")
                          + " hours; keep original plan, models, origins, seeds and stage caps.")
                session.extend_campaign_budget(requested_total - legacy_spent, reason=reason)
                status = deepcopy(previous)
                status.update(status="budget_extended_awaiting_resume", pause_reason=None,
                              budget=session.summary(),
                              combined_charged_seconds=legacy_spent + session.charged_wall_seconds)
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
                paths = tuple(sorted(set(session.pending_record_paths) | {"status.json"}))
                acknowledgement = mirror.push(output, paths=paths,
                    message="Register explicit user budget amendment before resuming fits")
                session.mark_records_synced(session.pending_record_paths)
                return {"status": "budget_amendment_confirmed", "previous_total_hours": current_total / 3600,
                        "total_hours": hours, "added_hours": (requested_total-current_total) / 3600,
                        "remaining_seconds": session.campaign_budget_seconds-session.charged_wall_seconds,
                        "previous_results_preserved": len(session.completed_job_ids),
                        "remote_commit": acknowledgement["remote_commit"], "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--total-hours", type=float, required=True)
    parser.add_argument("--confirmed-by-user", action="store_true")
    args = parser.parse_args()
    checked_hours(args.total_hours, args.confirmed_by_user)
    controller = args.controller_root.resolve()
    head = subprocess.run(["git", "-C", str(controller), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if head != args.controller_commit:
        raise ValueError("controller_checkout_pin_mismatch")
    changed = subprocess.run(["git", "-C", str(controller), "diff", "--name-only", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    extras = subprocess.run(["git", "-C", str(controller), "ls-files", "--others", "--exclude-standard", "scripts"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    if changed or any(name.endswith(".py") for name in extras):
        raise ValueError("controller_checkout_has_unregistered_changes")
    sys.path.insert(0, str(controller / "scripts"))
    opt = importlib.import_module("run_e2_optimized_selection_v2")
    opt.load_engine(args.engine_root)
    import e2_github_campaign as public
    from e2_github_git_transport import GitCampaignMirror
    from e2_search_session import SearchSession
    opt.install_public_schema(public)
    result = apply_extension(args, opt, GitCampaignMirror, SearchSession)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
