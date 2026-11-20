"""Fail CI when a Python dependency has a vulnerability on the CISA Known Exploited Vulnerabilities list."""

import json
import sys
import urllib.request

KEV_FEED = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def known_exploited(feed_url: str = KEV_FEED) -> set[str]:
    """CVE ids from the CISA KEV catalog."""
    with urllib.request.urlopen(feed_url, timeout=30) as response:
        catalog = json.load(response)
    return {entry["cveID"] for entry in catalog["vulnerabilities"]}


def findings(report_path: str) -> list[tuple[str, str, str, list[str]]]:
    """(package, version, advisory id, CVE aliases) for every advisory in a pip-audit JSON report."""
    with open(report_path) as handle:
        report = json.load(handle)
    rows = []
    for dependency in report.get("dependencies", []):
        for vulnerability in dependency.get("vulns", []):
            identifiers = [vulnerability["id"], *vulnerability.get("aliases", [])]
            rows.append(
                (
                    dependency["name"],
                    dependency.get("version", "?"),
                    vulnerability["id"],
                    [identifier for identifier in identifiers if identifier.startswith("CVE-")],
                )
            )
    return rows


def main(report_path: str) -> int:
    """Print every advisory and return 1 if any of them is known to be exploited."""
    rows = findings(report_path)
    exploited = known_exploited()
    blocking = [row for row in rows if exploited.intersection(row[3])]
    for name, version, identifier, cves in rows:
        marker = "KNOWN EXPLOITED" if exploited.intersection(cves) else "advisory"
        print(f"{marker}: {name} {version} {identifier} {' '.join(cves)}")
    print(f"{len(rows)} advisories, {len(blocking)} on the CISA known-exploited list")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
