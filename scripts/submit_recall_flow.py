#!/usr/bin/env python3
"""
Submit production recall workflow using the existing ads_prm package.

Follows the pattern from f1057460760 (blended_recall_eval.py):
- Uses FlowSession().schedule_workflow() to submit via Python
- Reuses the existing ads_prm package (no custom build needed)
- Sets skip_bulk_eval=True to use existing bulk eval data
- The PSelect recall workflow handles all pipeline stages

For agent recommendation, we reuse the same approach:
- The existing bulk eval table has PSelect predictions
- We schedule the recall workflow which runs the blending SQL
- At proportion=0.0, this gives us the production baseline
- To test agent scoring, we'd need to write agent scores to the
  bulk eval table first (future enhancement)

Usage (run via Bento or directly):
    python3 scripts/submit_recall_flow.py
    # Or in Bento: %run scripts/submit_recall_flow.py
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EVAL_DS = "2026-03-18"
PSELECT_MODEL_ID = 698799520  # production PSelect V0
SNAPSHOT_ID = 0
PAGE_TYPE = 19
NAMESPACE = "ad_delivery"

# Traffic proportions to evaluate
TRAFFIC_PROPORTIONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Package version (use published ads_prm package)
PACKAGE_VERSION = "ads_prm:187"


def submit_baseline_recall():
    """Submit baseline PSelect recall workflow (proportion=0.0 = no extra route)."""
    from fblearner.flow.external_api import FlowSession

    session = FlowSession()

    logger.info("Scheduling PSelect recall workflow (baseline)...")
    logger.info(f"  ds={EVAL_DS}, model={PSELECT_MODEL_ID}, page_type={PAGE_TYPE}")
    logger.info(f"  proportions={TRAFFIC_PROPORTIONS}")

    wf = session.schedule_workflow(
        workflow_name="ads.ads_prm.workflows_pselect_recall.recall_workflow",
        entitlement="ads_global_short_term_exploration",
        package_version=PACKAGE_VERSION,
        run_as_user=True,
        input_arguments={
            "ds_config": {"ds": EVAL_DS, "ds_start": None, "ds_end": None},
            "model_metadata": {
                "model_id": PSELECT_MODEL_ID,
                "snapshot_id": SNAPSHOT_ID,
            },
            "recall_type": "pm_to_ai_recall",
            "bulk_eval_config": {
                "prediction_column_name": "pselect_value",
                "output_table_name": None,
                "skip_bulk_eval": True,
                "entitlement": None,
                "secure_group": None,
                "use_model_dpo": False,
                "max_examples": -1,
            },
            "analysis_config": {
                "page_type": PAGE_TYPE,
                "extra_route_traffic_proportion_list": {
                    str(i): p for i, p in enumerate(TRAFFIC_PROPORTIONS)
                },
                "namespace": None,
                "disable_parallel_presto_execution": True,
            },
            "project_name": "agent_recommendation",
        },
    )

    run_id = wf.id
    logger.info(f"Workflow scheduled: f{run_id}")
    logger.info(f"Monitor: https://www.internalfb.com/intern/fblearner/details/{run_id}")

    # Save run info locally
    info = {
        "workflow_run_id": run_id,
        "ds": EVAL_DS,
        "model_id": PSELECT_MODEL_ID,
        "page_type": PAGE_TYPE,
        "proportions": TRAFFIC_PROPORTIONS,
        "url": f"https://www.internalfb.com/intern/fblearner/details/{run_id}",
    }
    out_path = Path(__file__).parent.parent / "evaluation" / "results" / "flow_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Run info saved to {out_path}")

    return run_id


def monitor_workflow(run_id: int, poll_interval: int = 60, max_wait: int = 3600):
    """Poll workflow status until completion."""
    from fblearner.flow.external_api import FlowSession

    session = FlowSession()
    start = time.time()

    while time.time() - start < max_wait:
        try:
            run = session.get_workflow_run(run_id)
            status = run.status
            elapsed = time.time() - start
            logger.info(f"  [{elapsed:.0f}s] f{run_id}: {status}")

            if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                if status == "SUCCEEDED":
                    logger.info("Workflow succeeded! Fetching outputs...")
                    try:
                        outputs = run.get_output()
                        logger.info(f"Outputs: {json.dumps(outputs, indent=2, default=str)[:2000]}")
                    except Exception as e:
                        logger.warning(f"Could not fetch outputs: {e}")
                return status
        except Exception as e:
            logger.warning(f"  Status check failed: {e}")

        time.sleep(poll_interval)

    logger.warning(f"Timed out after {max_wait}s")
    return "TIMEOUT"


def main():
    t0 = time.time()

    run_id = submit_baseline_recall()

    logger.info("\nMonitoring workflow...")
    status = monitor_workflow(run_id)

    elapsed = time.time() - t0
    logger.info(f"\nDone. Status: {status}, Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
