SPE_TERMS = {
    "spe",
    "spv",
    "entity",
    "vehicle",
    "partnership",
    "affiliate",
    "raptor",
    "ljm",
    "ljm1",
    "ljm2",
    "cheetah",
    "talon",
    "whitewing",
    "condor",
    "cayman",
}

SPE_PHRASES = [
    ("special", "purpose", "entity"),
    ("special", "purpose", "vehicle"),
    ("off", "balance", "sheet"),
    ("unconsolidated", "entity"),
]

DEBT_TERMS = {
    "hedge",
    "swap",
    "prepaid",
    "guarantee",
    "reserve",
    "impairment",
    "liquidity",
    "shortfall",
    "funding",
    "exposure",
    "risk",
}

DEBT_PHRASES = [
    ("prepaid", "transaction"),
    ("prepaid", "swap"),
    ("asset", "transfer"),
    ("liability", "transfer"),
    ("capital", "shortfall"),
]

ACCOUNTING_TERMS = {
    "valuation",
    "write",
    "default",
    "bankruptcy",
    "insolvency",
    "collapse",
    "restate",
    "adjustment",
    "audit",
    "auditor",
    "material",
    "consolidate",
    "transparency",
    "disclosure",
}

ACCOUNTING_PHRASES = [
    ("write", "down"),
    ("write", "off"),
    ("mark", "market"),
    ("fair", "value"),
]

RED_FLAG_TERMS = {
    "concern",
    "worry",
    "problem",
    "issue",
    "trouble",
    "volatile",
    "volatility",
}

ALL_TERMS = (
    SPE_TERMS
    | DEBT_TERMS
    | ACCOUNTING_TERMS
    | RED_FLAG_TERMS
)

ALL_PHRASES = (
    SPE_PHRASES
    + DEBT_PHRASES
    + ACCOUNTING_PHRASES
)
