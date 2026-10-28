"""SEC EDGAR filings + insider (Form 4) transactions (filings)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .. import config
from ..http import Http
from ..types import Company, RawItem
from ._util import finish, item_id, parse_iso

SOURCE = "sec"
URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
MAX_FORM4 = 10

FORMS = {"8-K", "10-Q", "10-K", "4", "S-1", "S-3", "424B5", "424B4", "SC 13D", "SC 13G", "SC 13D/A",
         "SCHEDULE 13D", "SCHEDULE 13G", "SCHEDULE 13D/A"}

ITEMS_8K = {
    "1.01": "Entry into a material definitive agreement",
    "1.02": "Termination of a material agreement",
    "1.03": "Bankruptcy or receivership",
    "2.01": "Completion of acquisition or disposition",
    "2.02": "Results of operations and financial condition",
    "2.03": "New direct financial obligation",
    "2.05": "Exit or restructuring costs",
    "2.06": "Material impairments",
    "3.01": "Notice of delisting",
    "3.02": "Unregistered sale of equity securities",
    "4.01": "Change of auditor",
    "4.02": "Non-reliance on previous financial statements",
    "5.01": "Change in control",
    "5.02": "Departure or appointment of officers or directors",
    "5.07": "Shareholder vote results",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
    "9.01": "Financial statements and exhibits",
}


def _title(form: str, items: str) -> str:
    if form == "8-K":
        descs = [ITEMS_8K.get(i.strip(), f"Item {i.strip()}") for i in items.split(",") if i.strip()]
        return "8-K: " + ("; ".join(descs) if descs else "Current report")
    if form in ("S-3", "424B5", "424B4"):
        return f"Shelf registration / securities offering prospectus ({form})"
    if form == "S-1":
        return "Registration statement (S-1)"
    if form.startswith(("SC 13D", "SCHEDULE 13D")):
        return "Activist stake disclosure (13D)"
    if form.startswith(("SC 13G", "SCHEDULE 13G")):
        return "Passive stake disclosure (13G)"
    if form == "10-Q":
        return "Quarterly report"
    if form == "10-K":
        return "Annual report"
    return form


def _filings(payload: dict, since: datetime) -> list[dict]:
    """Recent filings of the wanted forms, newest first, as flat dicts."""
    r = ((payload or {}).get("filings") or {}).get("recent") or {}
    out = []
    forms = r.get("form") or []
    for i, form in enumerate(forms):
        if form not in FORMS:
            continue
        try:
            acc = r["accessionNumber"][i]
            doc = r["primaryDocument"][i]
        except (KeyError, IndexError):
            continue
        dt = parse_iso((r.get("acceptanceDateTime") or [None] * len(forms))[i]) \
            or parse_iso((r.get("filingDate") or [None] * len(forms))[i])
        if not acc or not doc or dt is None or dt < since:
            continue
        items = (r.get("items") or [""] * len(forms))[i] or ""
        out.append({"form": form, "acc": acc, "doc": doc, "dt": dt, "items": items})
    return out


def _doc_url(company: Company, f: dict) -> str:
    return DOC_URL.format(cik=company.cik, acc=f["acc"].replace("-", ""), doc=f["doc"])


def _raw_xml_url(company: Company, f: dict) -> str:
    doc = f["doc"]
    parts = doc.split("/")
    if len(parts) > 1 and parts[0].lower().startswith("xsl"):
        doc = "/".join(parts[1:])
    return DOC_URL.format(cik=company.cik, acc=f["acc"].replace("-", ""), doc=doc)


def _money(v: float) -> str:
    v = abs(v)
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _num(node, path: str) -> float | None:
    el = node.find(path)
    if el is None or el.text is None:
        return None
    try:
        return float(el.text.strip())
    except ValueError:
        return None


def parse_form4(xml_text: str) -> dict | None:
    """-> {"owner","role","shares","value","price","verb"} or None when no P/S rows."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    owner_el = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
    owner = (owner_el.text or "").strip() if owner_el is not None and owner_el.text else ""
    role_el = root.find(".//reportingOwner/reportingOwnerRelationship/officerTitle")
    role = (role_el.text or "").strip() if role_el is not None and role_el.text else ""
    buy_sh = sell_sh = buy_val = sell_val = 0.0
    for tx in root.findall(".//nonDerivativeTransaction"):
        code_el = tx.find("transactionCoding/transactionCode")
        code = (code_el.text or "").strip() if code_el is not None and code_el.text else ""
        if code not in ("P", "S"):
            continue
        shares = _num(tx, "transactionAmounts/transactionShares/value")
        price = _num(tx, "transactionAmounts/transactionPricePerShare/value") or 0.0
        if not shares:
            continue
        if code == "P":
            buy_sh += shares
            buy_val += shares * price
        else:
            sell_sh += shares
            sell_val += shares * price
    if not (buy_sh or sell_sh):
        return None
    bought = buy_val >= sell_val if (buy_sh and sell_sh) else bool(buy_sh)
    shares = buy_sh if bought else sell_sh
    value = buy_val if bought else sell_val
    return {"owner": owner or "Unknown", "role": role, "shares": shares, "value": value,
            "price": value / shares if shares else 0.0, "verb": "bought" if bought else "sold",
            "net_value": buy_val - sell_val}


def _form4_item(company: Company, f: dict, xml_text: str) -> RawItem | None:
    d = parse_form4(xml_text)
    if d is None:
        return None
    who = f"{d['owner']} ({d['role']})" if d["role"] else d["owner"]
    title = f"Insider {who} {d['verb']} {d['shares']:,.0f} shares at ${d['price']:.2f} ({_money(d['value'])})"
    url = _doc_url(company, f)
    return RawItem(item_id(SOURCE, url, title), SOURCE, "filings", "SEC EDGAR", title, "", url, f["dt"],
                   company.ticker, {"form": "4", "insider_value": d["net_value"]})


def parse(payload: dict, company: Company, since: datetime,
          form4_xml: dict[str, str] | None = None) -> list[RawItem]:
    """form4_xml maps accession number -> raw XML; Form 4 filings without XML yield no item."""
    out = []
    for f in _filings(payload, since):
        if f["form"] == "4":
            if form4_xml and f["acc"] in form4_xml:
                it = _form4_item(company, f, form4_xml[f["acc"]])
                if it:
                    out.append(it)
            continue
        url = _doc_url(company, f)
        title = _title(f["form"], f["items"])
        out.append(RawItem(item_id(SOURCE, url, title), SOURCE, "filings", "SEC EDGAR", title, "", url,
                           f["dt"], company.ticker, {"form": f["form"]}))
    return finish(out, since)


def fetch(company: Company, http: Http, since: datetime) -> list[RawItem]:
    payload = http.get_json(URL.format(cik=company.cik), ua=config.SEC_UA)
    xmls: dict[str, str] = {}
    for f in [x for x in _filings(payload, since) if x["form"] == "4"][:MAX_FORM4]:
        try:
            xmls[f["acc"]] = http.get_text(_raw_xml_url(company, f), ua=config.SEC_UA)
        except Exception:
            continue
    return parse(payload, company, since, xmls)
