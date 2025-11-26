# AGAP Gurobi implementation (Sections 3 & 4) - revised
import gurobipy as gp
from gurobipy import GRB

def compute_NA_star(flights, gate_count_fixed, terminal_tag):
    """
    Computes minimum number of flights needing apron for one terminal.
    Uses a simpler time-interval model:
      max  sum u_i
      s.t. for each time interval r: sum_{i present} u_i ≤ gate_count_fixed
    NA* = total flights - max fixed-gate flights.
    """
    term_flights = [f for f in flights if f['terminal'] == terminal_tag]
    nF = len(term_flights)
    if nF == 0:
        return 0

    times = sorted({f['a'] for f in term_flights}.union({f['d'] for f in term_flights}))
    intervals = [(times[r], times[r+1]) for r in range(len(times)-1)]

    m = gp.Model(f"MaxFixed_{terminal_tag}")
    m.Params.OutputFlag = 0

    u = m.addVars([f['id'] for f in term_flights], vtype=GRB.BINARY, name="u")

    # Interval capacity constraints
    for r, (t0, t1) in enumerate(intervals):
        present = [f['id'] for f in term_flights if f['a'] < t1 and f['d'] > t0]
        if present:
            m.addConstr(gp.quicksum(u[i] for i in present) <= gate_count_fixed, name=f"cap_{r}")

    m.setObjective(gp.quicksum(u[i] for i in u.keys()), GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"NA* subproblem infeasible or not optimal (status {m.Status}).")

    Z_star = m.ObjVal
    return int(nF - Z_star)


def build_linear_AGAP_model(flights, p_matrix, gates_D, gates_I, apron_gate,
                            dist_gate_gate, dist_gate_entrance,
                            restrict_y_to_positive_p=True):
    """
    Builds linearized AGAP model (O2 with constraints (1)-(7)).
    """
    NA_D = compute_NA_star(flights, len(gates_D), 'D')
    NA_I = compute_NA_star(flights, len(gates_I), 'I')
    NA_star = NA_D + NA_I

    times = sorted({f['a'] for f in flights}.union({f['d'] for f in flights}))
    intervals = [(times[r], times[r+1]) for r in range(len(times)-1)]
    comp = {(f['id'], r): 1 if (f['a'] < t1 and f['d'] > t0) else 0
            for r, (t0, t1) in enumerate(intervals) for f in flights}

    K_D = gates_D + [apron_gate]
    K_I = gates_I + [apron_gate]

    m = gp.Model("Linear_AGAP")
    m.Params.OutputFlag = 0
    m.Params.LogFile = "agap_main.log"  # create solver log file

    x = {}
    for f in flights:
        gate_set = K_D if f['terminal'] == 'D' else K_I
        for k in gate_set:
            x[(f['id'], k)] = m.addVar(vtype=GRB.BINARY, name=f"x_{f['id']}_{k}")

    y = {}
    flights_sorted = sorted(flights, key=lambda z: z['id'])
    for i_idx, fi in enumerate(flights_sorted[:-1]):
        for fj in flights_sorted[i_idx+1:]:
            Ki = K_D if fi['terminal'] == 'D' else K_I
            Kj = K_D if fj['terminal'] == 'D' else K_I
            p_ij = p_matrix.get((fi['id'], fj['id']), 0)
            if restrict_y_to_positive_p and p_ij == 0:
                continue
            for k in Ki:
                for l in Kj:
                    y[(fi['id'], fj['id'], k, l)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                            name=f"y_{fi['id']}_{fj['id']}_{k}_{l}")

    # (1),(2)
    for f in flights:
        gate_set = K_D if f['terminal'] == 'D' else K_I
        m.addConstr(gp.quicksum(x[(f['id'], k)] for k in gate_set) == 1, name=f"assign_{f['id']}")

    # (3) no overlap on fixed gates (exclude apron)
    fixed_gates = list(set(gates_D + gates_I))
    for k in fixed_gates:
        for r, _ in enumerate(intervals):
            m.addConstr(
                gp.quicksum(comp[(f['id'], r)] * x[(f['id'], k)]
                            for f in flights
                            if (f['terminal'] == 'D' and k in K_D) or (f['terminal'] == 'I' and k in K_I)
                            if (f['id'], k) in x) <= 1,
                name=f"no_overlap_{k}_{r}"
            )

    # (4) apron usage fixed
    m.addConstr(gp.quicksum(x[(f['id'], apron_gate)] for f in flights
                            if (f['id'], apron_gate) in x) == NA_star, name="apron_count")

    # (6)
    for (i, j, k, l), yvar in y.items():
        m.addConstr(yvar >= x[(i, k)] + x[(j, l)] - 1, name=f"lin_{i}_{j}_{k}_{l}")

    # Objective (O2)
    obj = gp.LinExpr()
    for (i, j, k, l), yvar in y.items():
        p_ij = p_matrix.get((i, j), 0)
        obj.addTerms(p_ij * dist_gate_gate[(k, l)], yvar)

    for f in flights:
        gate_set = K_D if f['terminal'] == 'D' else K_I
        coeff = f['e'] + f['f']
        for k in gate_set:
            obj.addTerms(coeff * dist_gate_entrance[k], x[(f['id'], k)])

    m.setObjective(obj, GRB.MINIMIZE)
    m.update()

    return m, x, y, NA_star


if __name__ == "__main__":
    flights = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 5, 'e': 80, 'f': 75},
        {'id': 'F2', 'terminal': 'D', 'a': 2, 'd': 7, 'e': 60, 'f': 50},
        {'id': 'F3', 'terminal': 'I', 'a': 1, 'd': 6, 'e': 120, 'f': 110},
        {'id': 'F4', 'terminal': 'I', 'a': 6, 'd': 10, 'e': 90, 'f': 95},
    ]
    p_matrix = {
        ('F1', 'F2'): 30,
        ('F1', 'F3'): 10,
        ('F3', 'F4'): 40,
    }
    gates_D = ['D1', 'D2']
    gates_I = ['I1']
    apron_gate = 'APR'
    all_gates = gates_D + gates_I + [apron_gate]

    dist_gate_gate = {(g1, g2): (10 if g1 == g2 else 100) for g1 in all_gates for g2 in all_gates}
    dist_gate_entrance = {g: (20 if g != apron_gate else 400) for g in all_gates}

    model, x, y, NA_star = build_linear_AGAP_model(
        flights, p_matrix, gates_D, gates_I, apron_gate,
        dist_gate_gate, dist_gate_entrance
    )
    # Write LP before solve
    model.write("agap.lp")

    model.optimize()

    status = model.Status
    print("Model status:", status)
    if status == GRB.OPTIMAL:
        print("Objective value:", model.ObjVal)
        print("Minimum apron aircraft (NA*):", NA_star)
        for (fid, g) in x:
            if x[(fid, g)].X > 0.5:
                print(f"Flight {fid} -> Gate {g}")
        # Optional: write solution to a file
        model.write("agap.sol")
    else:
        if status == GRB.INFEASIBLE:
            print("Model infeasible. Computing IIS...")
            model.computeIIS()
            model.write("agap_iis.ilp")
        else:
            print("No optimal objective available.")