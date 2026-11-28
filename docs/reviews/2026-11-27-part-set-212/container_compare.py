import pathlib, re, sys, collections
SP = pathlib.Path(sys.argv[1])
res = {}
for v in ("branch", "main"):
    d = SP / f"cres-{v}"
    tests = {}; files = {}
    for line in "".join(p.read_text() for p in d.glob("summary_shard*")).splitlines():
        f, ex, sec = line.split()
        files[f] = (int(ex.split("=")[1]), int(sec.split("=")[1]))
    for f in files:
        log = (d / (f.replace("/", "_") + ".log")).read_text(errors="replace")
        for m in re.finditer(r"^(tests/\S+?::.+?) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b", log, re.M):
            tests[m.group(1)] = m.group(2)     # last word wins (teardown ERROR after PASSED)
        # collection-level errors / module skips
        if re.search(r"ERROR collecting", log): tests[f + "::<collection>"] = "ERROR"
        if files[f][0] == 124:
            seen = re.findall(r"^(tests/\S+?::\S+)", log, re.M)
            tests[f + "::<TIMEOUT after 600 s; last started: %s>" % (seen[-1] if seen else "?")] = "TIMEOUT"
    res[v] = (files, tests)
    c = collections.Counter(tests.values())
    print(f"{v:6s}: {len(files)} files  " + "  ".join(f"{k}={c[k]}" for k in sorted(c)))
    for f, (e, s) in sorted(files.items()):
        if e not in (0, 5): print(f"     exit {e:3d} {s:4d}s {f}")
b, m = res["branch"][1], res["main"][1]
bad = lambda t: {k for k, v in t.items() if v in ("FAILED", "ERROR", "TIMEOUT")}
only_b = sorted(bad(b) - bad(m)); only_m = sorted(bad(m) - bad(b))
print(f"\nfailing/erroring on the BRANCH but not on main: {len(only_b)}")
for k in only_b: print("  ", b[k], k[:170])
print(f"failing/erroring on MAIN but not on the branch: {len(only_m)}")
for k in only_m: print("  ", m[k], k[:170])
print(f"failing/erroring on both: {len(bad(b) & bad(m))}")
