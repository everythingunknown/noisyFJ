from pathlib import Path
import csv
import statistics
import tomllib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DATASETS = tomllib.loads((ROOT / "datasets.toml").read_text())["datasets"]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return "--" if value == "" else f"{float(value):.3g}"


def latex(name, headers, rows):
    lines = [r"\begin{tabular}{" + "l" + "r" * (len(headers)-1) + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    comparison, budgets, allocation, all_nodes, solvers = [], [], [], [], []
    for item in DATASETS:
        path = OUT / item["id"]
        if (path / "benchmark.complete").exists():
            row = {"network": item["name"], "n": item["n"], "m": item["m"],
                   "Exact_seconds": float(read(path / "exact.csv")[0]["seconds"]) if item["dense"] else ""}
            for method in ("FP", "AA"):
                queries = read(path / f"{method}.csv")
                row[f"{method}_seconds"] = statistics.mean(float(q["seconds"]) for q in queries)
                row[f"{method}_mae"] = statistics.mean(float(q["absolute_error"]) for q in queries) if item["dense"] else ""
            for result in read(path / "forests.csv"):
                budgets.append({"network": item["name"], **result})
                if result["forests"] == "4000":
                    row[result["method"] + "_seconds"] = float(result["total_s"])
                    row[result["method"] + "_mre_percent"] = float(result["mre_percent"])
            comparison.append(row)
        if (path / "all_nodes_summary.csv").exists():
            all_nodes.extend({"network": item["name"], **r} for r in read(path / "all_nodes_summary.csv"))
        if (path / "solver_baselines" / "complete").exists():
            solvers.extend({"network": item["name"], **r} for r in read(path / "solver_baselines" / "summary.csv"))
        if (path / "resampling" / "complete").exists():
            for result in read(path / "resampling" / "summary.csv"):
                allocation.append({"network": item["name"], **result})
    write("comparison.csv", comparison)
    write("forest_budgets.csv", budgets)
    write("resampling.csv", allocation)
    write("all_nodes_summary.csv", all_nodes)
    write("solver_baselines.csv", solvers)
    if comparison:
        latex("comparison.tex", ["Network", "$n$", "$m$", "Exact (s)", "FP (s)", "FP MAE",
              "AA (s)", "AA MAE", "PFS (s)", "PFS MRE (\\%)", "FRPFS (s)", "FRPFS MRE (\\%)"],
              [[r["network"], str(r["n"]), str(r["m"])] + [number(v) for v in list(r.values())[3:]]
               for r in comparison])
        latex("forest_budgets.tex", ["Network", "Forests", "Method", "Time (s)", "MAE", "$s_e$", "MRE (\\%)"],
              [[r["network"], r["forests"], r["method"]] +
               [number(r[k]) for k in ("total_s", "mae", "error_sd", "mre_percent")] for r in budgets])
    if allocation:
        plot(allocation)
    if all_nodes:
        plot_all_nodes()
    if solvers:
        index = {(r["network"], r["method"]): r for r in solvers}
        table = []
        for item in DATASETS:
            name = item["name"]
            if (name, "AA") not in index:
                continue
            aa, gmres, sor = [index[name, m] for m in ("AA", "GMRES", "SOR")]
            table.append([name, number(aa["total_s"]), number(gmres["total_s"]),
                          number(sor["total_s"]) + " (" + number(sor["omega"]) + ")"])
        latex("solver_baselines.tex", ["Network", "AA", "GMRES", r"SOR($\omega$)"], table)


def style():
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})


def plot(rows):
    style()
    names = [item["name"] for item in DATASETS if item["dense"]]
    methods = ["Degree", "Middle", "Uniform", "Structural", "Centrality"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    handles = labels = None
    for ax, name in zip(axes.flat, names):
        selected = {(r["method"], int(r["round"])): r for r in rows if r["network"] == name}
        if not selected:
            ax.set_visible(False)
            continue
        x = np.arange(len(methods))
        for round_no, offset, color in [(1, -0.18, "#4691d2"), (2, 0.18, "#e59b42")]:
            means = [float(selected[m, round_no]["mean_error"]) for m in methods]
            sds = [float(selected[m, round_no]["error_sd"]) for m in methods]
            ax.bar(x + offset, means, width=0.34, color=color, label=f"Round {round_no}")
            ax.errorbar(x + offset, means, yerr=sds, fmt="none", ecolor="#333333",
                        capsize=2, linewidth=0.7)
        ax.set_title(name)
        ax.set_xticks(x, methods, rotation=15)
        ax.set_ylabel("Squared equilibrium error")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "figure1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_nodes():
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    handles = labels = None
    for ax, item in zip(axes.flat, [d for d in DATASETS if d["dense"]]):
        path = OUT / item["id"] / "all_nodes.csv"
        if not path.exists():
            ax.set_visible(False)
            continue
        rows = read(path)
        rank = np.array([int(r["rank"]) for r in rows])
        exact = np.array([float(r["exact"]) for r in rows])
        estimate = np.array([float(r["frpfs"]) for r in rows])
        ax.plot(rank, exact, color="#4691d2", linewidth=1.2, label="Exact")
        ax.plot(rank, estimate, color="#32cd32", linewidth=0.8,
                linestyle=(0, (1.8, 3.2)), label="FRPFS")
        ax.set_title(item["name"])
        ax.set_xlabel("Node rank by exact centrality")
        ax.set_ylabel("Perturbation centrality")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "figure2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
