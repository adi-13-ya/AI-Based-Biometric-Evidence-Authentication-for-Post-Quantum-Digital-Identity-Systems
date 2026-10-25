"""Flatten every result file into report/data/v2/report_numbers.json.

The Word report (report/scripts/build_report.js) takes every number it
quotes from this file, so the text cannot drift from the measurements.
Percentages are stored as strings already formatted for the text.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "evaluation" / "results"
V2 = REPO / "report" / "data" / "v2"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def ms(x: float) -> str:
    return f"{x:.3f}" if x < 1 else f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def main() -> None:
    N: dict = {}
    pad = json.loads((RES / "pad_benchmark.json").read_text())
    for name, d in pad["datasets"].items():
        N[f"pad_{name}_n"] = d["images"]
        N[f"pad_{name}_bf"] = d["bona_fide"]
        N[f"pad_{name}_species"] = d["species"]
        N[f"pad_{name}_lowconf"] = pct(d["face_conf_below_0.6"])
    for sysname, res in pad["systems"].items():
        N[f"pad_{sysname}_dev_eer"] = pct(res["dev"]["eer"])
        N[f"pad_{sysname}_dev_auc"] = f(res["dev"]["auc"])
        N[f"pad_{sysname}_thr"] = f(res["dev"]["threshold"], 2)
        for ds, r in res["dev_eer"].items():
            k = f"pad_{sysname}_{ds}"
            N[k] = {
                "apcer_species": {s: pct(v) for s, v in r["apcer_per_species"].items()},
                "apcer_worst": pct(r["apcer_worst"]), "apcer_worst_species": r["apcer_worst_species"],
                "apcer_all": pct(r["apcer_all"]), "bpcer": pct(r["bpcer"]),
                "bpcer_ci": f"[{pct(r['bpcer_ci'][0])}, {pct(r['bpcer_ci'][1])}]",
                "apcer_ci": f"[{pct(r['apcer_all_ci'][0])}, {pct(r['apcer_all_ci'][1])}]",
                "auc": f(r["auc"]), "eer": pct(r["eer"]), "b5": pct(r["bpcer_at_apcer5"]), "b10": pct(r["bpcer_at_apcer10"]),
                "n_bf": r["n_bona_fide"], "n_att": r["n_attack"],
            }
    ft = json.loads((RES / "pad_finetune_log.json").read_text())
    N["pad_ft_best_epoch"] = ft.get("best_epoch")
    N["pad_ft_n_train"] = ft["n_train"]

    inj = json.loads((RES / "injection_eval.json").read_text())
    N["inj_videos"], N["inj_clips"] = inj["n_videos"], inj["n_clips"]
    for key in ("classifier", "legacy_naive_attacker", "legacy_adaptive_attacker"):
        r = inj[key]
        N[f"inj_{key}"] = {"bpcer": pct(r["bpcer"]), "auc": f(r["auc_trained_conditions"]), "eer": pct(r["eer_trained_conditions"]),
                           **{c: pct(v) for c, v in r["apcer_per_condition"].items()}}
    N["inj_threshold"] = f(inj["final_model"]["threshold_bpcer5"], 2)
    N["inj_coef"] = inj["final_model"]["coef"]

    if (RES / "deepfake_benchmark.json").exists():
        df = json.loads((RES / "deepfake_benchmark.json").read_text())
        N["df_datasets"] = df["datasets"]
        for sysname, r in df["systems"].items():
            N[f"df_{sysname}"] = {
                "dev_auc": f(r["dev_auc"]), "dev_eer": pct(r["dev_eer"]), "thr": f(r["threshold"], 2),
                "bpcer": pct(r["test"]["bpcer"]), "bpcer_ci": f"[{pct(r['test']['bpcer_ci'][0])}, {pct(r['test']['bpcer_ci'][1])}]",
                "auc": f(r["test"]["auc"]), "eer": pct(r["test"]["eer"]),
                "apcer": {g: pct(v) for g, v in r["test"]["apcer_per_species"].items()},
                "auc_gen": {g: f(v) for g, v in r["test_auc_per_generator"].items()},
            }
    if (RES / "deepfake_train_log.json").exists():
        log = json.loads((RES / "deepfake_train_log.json").read_text())
        N["df_best_epoch"], N["df_n_train"] = log["best_epoch"], log["n_train"]

    if (V2 / "d_summary.json").exists():
        s = json.loads((V2 / "d_summary.json").read_text())
        for mode, r in s.items():
            m = {"accept_rate": pct(r["genuine_accept_rate"]), "verify_med": ms(r["genuine_verify_http_ms"]["median"]),
                 "verify_p95": ms(r["genuine_verify_http_ms"]["p95"]), "rss_start": f"{r['rss_mb_start']:.0f}", "rss_after": f"{r['rss_mb_after_genuine']:.0f}",
                 "flood_rate": f"{r['flood']['forged_per_second']:.1f}", "flood_n": r["flood"]["forged_requests"],
                 "flood_genuine_n": r["flood"]["genuine_completed"],
                 "flood_genuine_med": ms(r["flood"]["genuine_verify_ms"]["median"]) if r["flood"]["genuine_verify_ms"] else "n/a"}
            for kind, k in r["reject"].items():
                m[kind] = {"http": ms(k["verify_http_ms"]["median"]), "server": ms(k["server_total_ms"]["median"]),
                           "cpu": ms(k["server_cpu_ms"]["median"]), "detection_ran": k["detection_ran"],
                           "all_rejected": k["all_rejected"], "check": ",".join(k["failed_check"]),
                           "cpu_raw": k["server_cpu_ms"]["median"], "server_raw": k["server_total_ms"]["median"]}
            N[f"sys_{mode}"] = m
        g = list(csv.DictReader(open(V2 / "d1_genuine_integrated.csv")))
        med = lambda key: statistics.median(float(r[key]) for r in g if r.get(key) not in (None, ""))  # noqa: E731
        stages = ["session_start_ms", "encode_frames_ms", "hash_ms", "sign_ms", "encaps_seal_ms", "verify_http_ms",
                  "server_kem_decapsulate_ms", "server_aead_decrypt_ms", "server_binding_check_ms", "server_liveness_check_ms",
                  "server_detection_ms", "server_fusion_ms", "server_total_ms", "payload_bytes"]
        N["e2e"] = {k: (ms(med(k)) if k != "payload_bytes" else f"{med(k) / 1024:.0f} KiB") for k in stages}
        N["e2e_raw"] = {k: med(k) for k in stages}
        pqc = med("server_kem_decapsulate_ms") + med("server_aead_decrypt_ms") + med("server_binding_check_ms")
        N["pqc_share"] = pct(pqc / med("server_total_ms"))
        N["pqc_server_ms"] = ms(pqc)
        N["client_crypto_ms"] = ms(med("hash_ms") + med("sign_ms") + med("encaps_seal_ms"))
        N["n_genuine"] = len(g)
        integ, lay = s["integrated"]["reject"], s["layered"]["reject"]
        N["dos_ratio_forged"] = f"{lay['forged_signature']['server_total_ms']['median'] / max(1e-9, integ['forged_signature']['server_total_ms']['median']):.0f}"
    if (V2 / "d3_flood.json").exists():
        for r in json.loads((V2 / "d3_flood.json").read_text()):
            key = f"flood_{r['mode']}_{'on' if r['flood'] else 'off'}"
            N[key] = {"victim_med": f"{r['victim_ms_median'] / 1000:.2f}", "victim_p95": f"{r['victim_ms_p95'] / 1000:.2f}",
                      "victim_n": r["victim_completed"], "victim_acc": r["victim_accepted"], "forged_rate": f"{r['forged_per_s']:.1f}",
                      "forged_n": r["forged_requests"], "forged_acc": r["forged_accepted"], "seconds": r["seconds"],
                      "attackers": r["attackers"], "victim_rate": f"{r['victim_completed'] / r['victim_active_s']:.2f}",
                      "victim_med_raw": r["victim_ms_median"]}
        N["flood_slowdown_integrated"] = f"{N['flood_integrated_on']['victim_med_raw'] / N['flood_integrated_off']['victim_med_raw']:.1f}"
        N["flood_slowdown_layered"] = f"{N['flood_layered_on']['victim_med_raw'] / N['flood_layered_off']['victim_med_raw']:.1f}"
        N["flood_absorb_ratio"] = f"{float(N['flood_integrated_on']['forged_rate']) / max(1e-9, float(N['flood_layered_on']['forged_rate'])):.1f}"
    if (V2 / "d4_module_latency.csv").exists():
        N["modules"] = list(csv.DictReader(open(V2 / "d4_module_latency.csv")))
    if (V2 / "s1_primitives.csv").exists():
        N["primitives"] = list(csv.DictReader(open(V2 / "s1_primitives.csv")))

    q = json.loads((REPO / "report" / "data" / "quantum_run_log.json").read_text())
    N["quantum_log_keys"] = list(q)[:20]
    (V2).mkdir(parents=True, exist_ok=True)
    (V2 / "report_numbers.json").write_text(json.dumps(N, indent=1))
    print(f"{len(N)} keys -> {V2 / 'report_numbers.json'}")


if __name__ == "__main__":
    main()
