"""Planning arithmetic in USD for the Azure deployment: not a quote and not a spending control.

Rates come from deploy/pricing-reference.json (Azure Retail Prices API, list prices). The VM is
charged while it runs; the OS disk and public IP are charged from creation until they are deleted,
including while the VM is deallocated.

    python deploy/estimate_costs.py --service-hours 48 --setup-hours 8 --teardown-hours 16
"""
import argparse
import json
from pathlib import Path

REF = json.loads((Path(__file__).with_name("pricing-reference.json")).read_text(encoding="utf-8"))
SAR_PER_USD = 3.75  # fixed peg


def estimate(service_hours, setup_hours, teardown_hours, outbound_gb, vm_rate):
    vm_hours = setup_hours + service_hours
    held_hours = vm_hours + teardown_hours  # disk and IP exist until the resource group is deleted
    disk_hourly = REF["os_disk"]["retailPrice"] / REF["billing_hours_per_month_assumed"]
    lines = {
        f"VM {vm_hours} h x {vm_rate}": vm_hours * vm_rate,
        f"OS disk {held_hours} h x {disk_hourly:.5f}": held_hours * disk_hourly,
        f"public IP {held_hours} h x {REF['public_ip']['retailPrice']}": held_hours * REF["public_ip"]["retailPrice"],
        f"outbound {outbound_gb} GB x {REF['outbound_data_usd_per_gb_assumed']}": outbound_gb * REF["outbound_data_usd_per_gb_assumed"],
        "shutdown automation reserve": 0.10,
    }
    return lines, sum(lines.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-hours", type=float, default=48)
    ap.add_argument("--setup-hours", type=float, default=8, help="VM running before the public window (install, store upload, checks)")
    ap.add_argument("--teardown-hours", type=float, default=16, help="VM deallocated but disk/IP not yet deleted")
    ap.add_argument("--outbound-gb", type=float, default=10)
    ap.add_argument("--credit-usd", type=float, default=200)
    ap.add_argument("--vm", choices=["D4ds_v4"], default="D4ds_v4")
    a = ap.parse_args()
    rate = REF["vm"]["retailPrice"]
    lines, total = estimate(a.service_hours, a.setup_hours, a.teardown_hours, a.outbound_gb, rate)
    for k, v in lines.items():
        print(f"{k:<45} {v:8.2f}")
    print(f"{'total USD':<45} {total:8.2f}  (SAR {total * SAR_PER_USD:.2f})")
    print(f"{'free credit USD':<45} {a.credit_usd:8.2f}  remaining {a.credit_usd - total:.2f} ({(a.credit_usd - total) / a.credit_usd:.0%})")
