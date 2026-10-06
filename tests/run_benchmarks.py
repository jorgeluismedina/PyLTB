"""
Corre los benchmarks de PyLTB y compara sus salidas con las de otra corrida.

    python tests/run_benchmarks.py --out base.json --ref HEAD     # pyltb de HEAD, tests actuales
    python tests/run_benchmarks.py --out nuevo.json --compare base.json
    python tests/run_benchmarks.py --compare base.json --against nuevo.json   # sin volver a correr

Corre cada tests/*/*.py en un subproceso con el mismo intérprete y MPLBACKEND=Agg (sin ventanas), con
un tiempo límite, y guarda su stdout completo. Excluye convergence_tests/, que sobrescribe las figuras
de la tesis. Con --ref, los tests son los del árbol de trabajo y pyltb es el de ese commit (git archive
en una carpeta temporal). --compare extrae los números línea por línea e informa, por script, la mayor
diferencia relativa y las líneas que cambiaron. Se corre desde la raíz del repositorio y no escribe en
tests/. El redondeo de μ_cr llega a ~1e-5 %: una diferencia de ese orden no es un cambio.
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NUM  = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def scripts(only):
    """tests/*/*.py sin convergence_tests/, filtrados por los patrones de --only."""
    found = sorted(p for p in (REPO / "tests").glob("*/*.py") if p.parent.name != "convergence_tests")
    rel = [p.relative_to(REPO / "tests").as_posix() for p in found]
    return [r for r in rel if not only or any(o in r for o in only)]


def pyltb_from(ref, tmp):
    """Extrae el pyltb del commit ref en tmp (git archive en zip)."""
    zipped = subprocess.run(["git", "archive", "--format=zip", ref, "pyltb"], cwd=REPO,
                            capture_output=True, check=True).stdout
    zipfile.ZipFile(io.BytesIO(zipped)).extractall(tmp)


def run_all(names, env, timeout):
    out = {}
    for name in names:
        t = time.time()
        try:
            r = subprocess.run([sys.executable, str(REPO / "tests" / name)], cwd=REPO, env=env,
                               capture_output=True, text=True, encoding="utf-8", timeout=timeout)
            rc, stdout, stderr = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = None, "", f"tiempo límite de {timeout} s"
        out[name] = {"rc": rc, "seconds": round(time.time() - t, 1), "stdout": stdout, "stderr": stderr}
        print(f"  {name:<45} rc={rc}  {out[name]['seconds']:6.1f} s", flush=True)
    return out


def rel_diff(a, b):
    a, b = float(a), float(b)
    return abs(b - a) / max(abs(a), abs(b), 1e-300)


def compare(base, new, tol, max_lines):
    """Informe por script: igual / CAMBIA / error, la mayor diferencia relativa y las líneas que cambian."""
    count = {"igual": 0, "CAMBIA": 0, "error": 0}
    print(f"\n{'script':<45} {'estado':<8} {'máx. dif. %':>12}")
    for name in sorted(set(base) | set(new)):
        a, b = base.get(name), new.get(name)
        if a is None or b is None:
            print(f"{name:<45} {'solo en ' + ('nuevo' if a is None else 'base'):<20}")
            count["error"] += 1
            continue
        if a["rc"] != 0 or b["rc"] != 0:
            print(f"{name:<45} {'error':<8} rc base={a['rc']} nuevo={b['rc']}")
            tail = (b["stderr"] if b["rc"] != 0 else a["stderr"]).strip().splitlines()[-3:]
            print("".join(f"    {s}\n" for s in tail), end="")
            count["error"] += 1
            continue
        la, lb = a["stdout"].splitlines(), b["stdout"].splitlines()
        changed, worst, text_only = [], 0.0, False
        if len(la) != len(lb):
            print(f"{name:<45} {'CAMBIA':<8} {'':>12}  ({len(la)} líneas -> {len(lb)})")
            count["CAMBIA"] += 1
            continue
        for i, (x, y) in enumerate(zip(la, lb), 1):
            if x == y:
                continue
            nx, ny = NUM.findall(x), NUM.findall(y)
            if len(nx) != len(ny):
                changed.append((i, x, y)); text_only = True
                continue
            d = max((rel_diff(p, q) for p, q in zip(nx, ny) if p != q), default=0.0)
            if d > tol:
                changed.append((i, x, y))
            worst = max(worst, d)
        state = "CAMBIA" if changed else "igual"
        count[state] += 1
        print(f"{name:<45} {state:<8} {100 * worst:>12.3g}" + ("  (texto distinto)" if text_only else ""))
        for i, x, y in changed[:max_lines]:
            print(f"    l. {i}: {x.strip()}\n    {'':>{len(str(i)) + 4}}{y.strip()}")
        if len(changed) > max_lines:
            print(f"    ... y {len(changed) - max_lines} líneas más")
    print(f"\n{count['igual']} iguales, {count['CAMBIA']} cambian, {count['error']} con error"
          f" (tolerancia {100 * tol:g} %)")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--out", help="JSON donde guardar las salidas de esta corrida")
    ap.add_argument("--ref", help="commit de pyltb con el que correr los tests (p. ej. HEAD, main)")
    ap.add_argument("--compare", help="JSON de una corrida anterior con el que comparar")
    ap.add_argument("--against", help="JSON a comparar con --compare, en vez de correr los tests")
    ap.add_argument("--only", nargs="*", default=[], help="correr solo los scripts que contengan estos textos")
    ap.add_argument("--timeout", type=float, default=300, help="tiempo límite por script [s]")
    ap.add_argument("--tol", type=float, default=1e-7, help="diferencia relativa que no cuenta como cambio")
    ap.add_argument("--max-lines", type=int, default=12, help="líneas cambiadas que se muestran por script")
    args = ap.parse_args()
    if isinstance(sys.stdout, io.TextIOWrapper):                  # los resúmenes usan caracteres de recuadro
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.against:
        new = json.loads(Path(args.against).read_text(encoding="utf-8"))["scripts"]
    else:
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        tmp = None
        if args.ref:
            tmp = tempfile.mkdtemp(prefix="pyltb_ref_")
            pyltb_from(args.ref, tmp)
            env["PYTHONPATH"] = tmp + os.pathsep + env.get("PYTHONPATH", "")
            # desde tests/, como los scripts: en la raíz, `import pyltb` encuentra la carpeta local
            probe = subprocess.run([sys.executable, "-c", "import pyltb; print(pyltb.__file__)"],
                                   cwd=REPO / "tests", env=env, capture_output=True, text=True).stdout.strip()
            if not Path(probe).resolve().is_relative_to(Path(tmp).resolve()):
                sys.exit(f"pyltb no se importa desde {tmp} sino desde {probe}")
        names = scripts(args.only)
        print(f"{len(names)} scripts con el pyltb " + (f"de {args.ref}:" if args.ref else "del árbol de trabajo:"))
        try:
            new = run_all(names, env, args.timeout)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"ref": args.ref or "árbol de trabajo", "date": time.strftime("%Y-%m-%d %H:%M"),
                 "python": sys.executable, "scripts": new}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Salidas en {args.out}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))["scripts"]
        compare(base, new, args.tol, args.max_lines)


if __name__ == "__main__":
    main()
