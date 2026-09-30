# tests_offline/test_data_mock.py
# Pure sanity checks on the fixture domain. Exists so a future edit to
# data.py can't silently let a training ticket and an eval ticket share
# an id (which would blur the "distilled from train, scored on held-out
# eval" separation this repo's whole comparison depends on), or let an
# eval ticket's expected label fall outside the valid set.

from scenario_ticket_extraction.data import TRAIN_TICKETS, EVAL_TICKETS, VALID_CATEGORIES, VALID_SEVERITIES


def main():
    train_ids = {t["ticket_id"] for t in TRAIN_TICKETS}
    eval_ids = {t["ticket_id"] for t in EVAL_TICKETS}
    assert len(train_ids) == len(TRAIN_TICKETS), "duplicate ticket_id in TRAIN_TICKETS"
    assert len(eval_ids) == len(EVAL_TICKETS), "duplicate ticket_id in EVAL_TICKETS"
    assert train_ids.isdisjoint(eval_ids), "train and eval sets must never share a ticket_id"
    print("  [no duplicate or overlapping ticket_id between train and eval sets] OK")

    for t in EVAL_TICKETS:
        assert t["expected_category"] in VALID_CATEGORIES, f"{t['ticket_id']} has unknown expected_category"
        assert t["expected_severity"] in VALID_SEVERITIES, f"{t['ticket_id']} has unknown expected_severity"
    print("  [every eval ticket's expected labels are in the valid sets] OK")

    assert len(TRAIN_TICKETS) >= 10, "expected enough training tickets for a meaningful LoRA fine-tune"
    assert len(EVAL_TICKETS) >= 6, "expected enough eval tickets for a meaningful evaluation"
    print("  [enough tickets in both sets for a meaningful train/eval split] OK")

    print("data offline validation: all checks passed")


if __name__ == "__main__":
    main()
