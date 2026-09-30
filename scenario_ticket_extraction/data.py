# scenario_ticket_extraction/data.py
# The task: turn a free-text support ticket into a fixed JSON schema,
# {"category": "billing"|"technical"|"account", "severity": "low"|"medium"|"high",
# "root_cause_guess": a short phrase}. Continues the ops/support-ticket
# theme from MCP and LLM Internals, a deliberate choice, not a
# coincidence: readers following the series already know this domain,
# so the fine-tuning story can focus entirely on the new variable
# (model size and tuning), not on re-explaining a new scenario.
#
# TRAIN_TICKETS have no reference labels on purpose. This repo's real
# training data comes from Gemini's own zero-shot output on these
# tickets (see scenario_ticket_extraction/distill.py), a genuine,
# current enterprise pattern: distill a large hosted model's behavior
# on a narrow task into a small model you can run yourself. EVAL_TICKETS
# are held out from training entirely and carry real, hand-assigned
# expected labels, so every one of the 3 systems this repo compares is
# scored against the same, independent ground truth.

VALID_CATEGORIES = ["billing", "technical", "account"]
VALID_SEVERITIES = ["low", "medium", "high"]

TRAIN_TICKETS = [
    {"ticket_id": "TR1", "text": "I was billed twice for my annual plan renewal, please refund the duplicate charge."},
    {"ticket_id": "TR2", "text": "Every time I try to export a report the page freezes and I have to reload the whole app."},
    {"ticket_id": "TR3", "text": "I forgot my password and the reset email never arrived, checked spam too."},
    {"ticket_id": "TR4", "text": "The checkout page is completely down, none of our customers can complete a purchase right now."},
    {"ticket_id": "TR5", "text": "Can I get an itemized breakdown of last month's invoice? A few line items look off."},
    {"ticket_id": "TR6", "text": "The mobile app crashes immediately on launch since the latest update."},
    {"ticket_id": "TR7", "text": "I'd like to add a second admin to our account, where do I manage team permissions?"},
    {"ticket_id": "TR8", "text": "Our API integration started returning 500 errors on every request about an hour ago, this is blocking our production pipeline."},
    {"ticket_id": "TR9", "text": "Is there a way to switch from monthly to annual billing without losing my current discount?"},
    {"ticket_id": "TR10", "text": "The dark mode toggle in settings doesn't actually change anything, minor annoyance but wanted to flag it."},
    {"ticket_id": "TR11", "text": "I keep getting logged out every few minutes, it's making the product unusable for my whole team today."},
    {"ticket_id": "TR12", "text": "Requesting a copy of last year's invoices for our accounting team's records."},
    {"ticket_id": "TR13", "text": "The search bar returns a blank page instead of results, been happening since this morning."},
    {"ticket_id": "TR14", "text": "How do I close my account entirely and make sure recurring billing stops?"},
]

EVAL_TICKETS = [
    {"ticket_id": "EV1", "text": "My card was charged the wrong amount this cycle, it should have been the discounted rate.", "expected_category": "billing", "expected_severity": "medium"},
    {"ticket_id": "EV2", "text": "The entire dashboard has been returning a 502 error for the last 30 minutes, nobody on my team can log in.", "expected_category": "technical", "expected_severity": "high"},
    {"ticket_id": "EV3", "text": "I can't find the option to change my display name in account settings.", "expected_category": "account", "expected_severity": "low"},
    {"ticket_id": "EV4", "text": "We were double-invoiced for 2 consecutive months, need both duplicate charges reversed.", "expected_category": "billing", "expected_severity": "high"},
    {"ticket_id": "EV5", "text": "Uploading a file larger than 10MB silently fails with no error message.", "expected_category": "technical", "expected_severity": "medium"},
    {"ticket_id": "EV6", "text": "Please remove a former employee's access from our shared workspace.", "expected_category": "account", "expected_severity": "medium"},
    {"ticket_id": "EV7", "text": "The billing page shows a past-due balance even though autopay went through fine last week.", "expected_category": "billing", "expected_severity": "medium"},
    {"ticket_id": "EV8", "text": "Notifications stopped arriving entirely, not a huge deal day to day but I'd like it fixed eventually.", "expected_category": "technical", "expected_severity": "low"},
]
