# AGAP Gurobi implementation (Sections 3 & 4) - revised
import gurobipy as gp
from gurobipy import GRB




def compute_NA_star(flights, gate_count_fixed, terminal_tag):
    """
    Computes minimum number of flights needing apron for one terminal.
    
    Uses the maximum cost network flow model from the paper (Section 3):
    - Nodes: source (0), aircraft (1 to |I|), sink (|I|+1)
    - Arcs A: (0,j) for all aircraft j, (i,|I|+1) for all aircraft i,
              and (i,j) between non-overlapping aircraft pairs
    - All arc weights = 1
    - Decision variable: z_ij = 1 if arc (i,j) is selected
    
    Mathematical formulation:
        Max Z = sum_{(i,j) in A} z_ij                           (maximize arcs visited)
        s.t.
        sum_{(0,j) in A} z_0j <= |K| - 1                        (flow from source)
        sum_{(i,|I|+1) in A} z_{i,|I|+1} <= |K| - 1             (flow to sink)
        sum_{(i,j) in A} z_ij = sum_{(j,i) in A} z_ji, i=1..|I| (flow conservation)
        sum_{(i,j) in A} z_ij <= 1, i=1..|I|                    (each aircraft visited once)
        z_ij in {0,1}
    
    Key insight from paper: K includes apron, so |K| - 1 = number of FIXED gates.
    Since gate_count_fixed already represents fixed gates only, we use it directly.
    
    Each path from source to sink represents a chain of non-overlapping aircraft
    that can be assigned to ONE fixed gate. With |K|-1 paths (= fixed gate count),
    we cover the maximum aircraft using all fixed gates.
    
    NA* (for this terminal) = |I| - (number of aircraft covered by paths)
    
    Args:
        flights: list of flight dicts
        gate_count_fixed: number of fixed gates (NOT including apron)
        terminal_tag: 'D' or 'I'
    """
    term_flights = [f for f in flights if f['terminal'] == terminal_tag]
    n_flights = len(term_flights)
    
    if n_flights == 0:
        return 0
    
    # Node indices: 0 = source, 1..n_flights = aircraft, n_flights+1 = sink
    source = 0
    sink = n_flights + 1
    aircraft_nodes = list(range(1, n_flights + 1))
    
    # Map flight id to node index and vice versa
    flight_to_node = {term_flights[i]['id']: i + 1 for i in range(n_flights)}
    node_to_flight = {i + 1: term_flights[i] for i in range(n_flights)}
    
    # Build arc set A_D (or A_I for international)
    # Arcs: (source, j) for all aircraft j
    #       (i, sink) for all aircraft i  
    #       (i, j) for non-overlapping pairs where i < j (to avoid duplicates)
    arcs = []
    
    # Arcs from source to each aircraft
    for j in aircraft_nodes:
        arcs.append((source, j))
    
    # Arcs from each aircraft to sink
    for i in aircraft_nodes:
        arcs.append((i, sink))
    
    # Arcs between non-overlapping aircraft pairs
    # Arc (i,j) exists only if flight i can be followed by flight j at the same gate
    # This requires: fi departs before or when fj arrives (fi['d'] <= fj['a'])
    for i in aircraft_nodes:
        for j in aircraft_nodes:
            if i != j:
                fi = node_to_flight[i]
                fj = node_to_flight[j]
                # Arc from i to j only if fi finishes before/when fj starts
                if fi['d'] <= fj['a']:
                    arcs.append((i, j))
    
    # Build the optimization model
    m = gp.Model(f"MaxCostFlow_{terminal_tag}")
    m.Params.OutputFlag = 0
    
    # Decision variables: z_ij for each arc
    z = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}") for (i, j) in arcs}
    
    # Constraint: flow out of source <= |K| - 1 = gate_count_fixed
    # (K includes apron, so |K|-1 = number of fixed gates)
    source_out_arcs = [(i, j) for (i, j) in arcs if i == source]
    m.addConstr(
        gp.quicksum(z[arc] for arc in source_out_arcs) <= gate_count_fixed,
        name="source_flow"
    )
    
    # Constraint: flow into sink <= |K| - 1 = gate_count_fixed
    sink_in_arcs = [(i, j) for (i, j) in arcs if j == sink]
    m.addConstr(
        gp.quicksum(z[arc] for arc in sink_in_arcs) <= gate_count_fixed,
        name="sink_flow"
    )
    
    # Constraint: flow conservation at each aircraft node
    # sum of incoming arcs = sum of outgoing arcs
    for node in aircraft_nodes:
        in_arcs = [(i, j) for (i, j) in arcs if j == node]
        out_arcs = [(i, j) for (i, j) in arcs if i == node]
        m.addConstr(
            gp.quicksum(z[arc] for arc in in_arcs) == gp.quicksum(z[arc] for arc in out_arcs),
            name=f"flow_conserve_{node}"
        )
    
    # Constraint: each aircraft can be visited at most once (outflow <= 1)
    for node in aircraft_nodes:
        out_arcs = [(i, j) for (i, j) in arcs if i == node]
        m.addConstr(
            gp.quicksum(z[arc] for arc in out_arcs) <= 1,
            name=f"visit_once_{node}"
        )
    
    # Objective: maximize total arcs selected (all weights = 1)
    m.setObjective(gp.quicksum(z[arc] for arc in arcs), GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"NA* subproblem infeasible or not optimal (status {m.Status}).")
    
    Z_star = m.ObjVal
    
    # The paper states: max aircraft assignable to |K|-1 gates = Z*
    # So NA* = n_flights - (aircraft covered by the flow)
    # Note: Z* counts arcs, but aircraft covered = arcs through aircraft nodes
    # Each aircraft visited contributes 2 arcs (in + out), plus source/sink arcs
    # Actually, aircraft assigned = number of aircraft nodes with flow through them
    
    # Count aircraft that are part of a path (have outgoing flow)
    aircraft_assigned = sum(1 for node in aircraft_nodes 
                           if any(z[(i, j)].X > 0.5 for (i, j) in arcs if i == node))
    
    return int(n_flights - aircraft_assigned)


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
    base_flights = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 60, 'e': 150, 'f': 140},
        {'id': 'F2', 'terminal': 'D', 'a': 20, 'd': 80, 'e': 120, 'f': 110},
        {'id': 'F3', 'terminal': 'D', 'a': 50, 'd': 110, 'e': 100, 'f': 95},  # Overlaps with F1 and F2 at t=50-60
        {'id': 'F4', 'terminal': 'D', 'a': 90, 'd': 160, 'e': 80, 'f': 75},
        {'id': 'F5', 'terminal': 'I', 'a': 0, 'd': 50, 'e': 200, 'f': 180},
        {'id': 'F6', 'terminal': 'I', 'a': 30, 'd': 90, 'e': 160, 'f': 150},  # Overlaps with F5 at t=30-50
        {'id': 'F7', 'terminal': 'I', 'a': 60, 'd': 120, 'e': 140, 'f': 130},  # Overlaps with F6 at t=60-90
        {'id': 'F8', 'terminal': 'I', 'a': 100, 'd': 180, 'e': 110, 'f': 100},
    ]
    
    base_p_matrix = {
        ('F1', 'F2'): 25, ('F1', 'F3'): 15, ('F1', 'F5'): 30,
        ('F2', 'F3'): 20, ('F2', 'F4'): 10,
        ('F3', 'F4'): 25,
        ('F5', 'F6'): 35, ('F5', 'F7'): 20,
        ('F6', 'F7'): 30, ('F6', 'F8'): 15,
        ('F7', 'F8'): 25,
    }
    
    base_gates_D = ['D1', 'D2']
    base_gates_I = ['I1', 'I2']
    apron_gate = 'APR'
    all_gates = base_gates_D + base_gates_I + [apron_gate]

    dist_gate_gate = {(g1, g2): (10 if g1 == g2 else 100) for g1 in all_gates for g2 in all_gates}
    dist_gate_entrance = {g: (20 if g != apron_gate else 400) for g in all_gates}

    model, x, y, NA_star = build_linear_AGAP_model(
        base_flights, base_p_matrix, base_gates_D, base_gates_I, apron_gate,
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