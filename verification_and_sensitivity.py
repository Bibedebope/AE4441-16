# Verification and Sensitivity Analysis for AGAP Model
import gurobipy as gp
from gurobipy import GRB
from agap import compute_NA_star, build_linear_AGAP_model, flights_overlap
import json
from datetime import datetime

def run_verification_tests():
    """
    Run verification tests to ensure model constraints work correctly.
    Returns a dictionary of test results.
    """
    results = {}
    
    # ==========================================================================
    # TEST 1: No overlap constraint - two overlapping flights, one gate
    # Expected: One flight must go to apron
    # ==========================================================================
    print("=" * 60)
    print("TEST 1: Overlap constraint with insufficient gates")
    print("=" * 60)
    
    flights_t1 = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 10, 'e': 100, 'f': 100},
        {'id': 'F2', 'terminal': 'D', 'a': 5, 'd': 15, 'e': 100, 'f': 100},  # Overlaps with F1
    ]
    gates_D_t1 = ['D1']  # Only 1 gate for 2 overlapping flights
    gates_I_t1 = []
    apron = 'APR'
    all_gates_t1 = gates_D_t1 + gates_I_t1 + [apron]
    
    dist_gg_t1 = {(g1, g2): 100 for g1 in all_gates_t1 for g2 in all_gates_t1}
    dist_ge_t1 = {g: 50 for g in all_gates_t1}
    p_matrix_t1 = {}
    
    NA_star_t1 = compute_NA_star(flights_t1, len(gates_D_t1), 'D')
    print(f"  Flights: F1 [0-10], F2 [5-15] (overlapping)")
    print(f"  Gates: 1 domestic")
    print(f"  Computed NA*: {NA_star_t1}")
    print(f"  Expected NA*: 1 (one flight must use apron)")
    
    test1_pass = (NA_star_t1 == 1)
    results['test1_overlap_constraint'] = {
        'description': 'Two overlapping flights with one gate',
        'NA_star': NA_star_t1,
        'expected': 1,
        'passed': test1_pass
    }
    print(f"  TEST 1: {'PASSED' if test1_pass else 'FAILED'}")
    
    # ==========================================================================
    # TEST 2: Non-overlapping flights can share a gate
    # Expected: NA* = 0 (all fit on fixed gates)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Non-overlapping flights share gate")
    print("=" * 60)
    
    flights_t2 = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 5, 'e': 100, 'f': 100},
        {'id': 'F2', 'terminal': 'D', 'a': 5, 'd': 10, 'e': 100, 'f': 100},  # Starts when F1 ends
        {'id': 'F3', 'terminal': 'D', 'a': 10, 'd': 15, 'e': 100, 'f': 100},  # Sequential
    ]
    gates_D_t2 = ['D1']  # 1 gate for 3 sequential flights
    
    NA_star_t2 = compute_NA_star(flights_t2, len(gates_D_t2), 'D')
    print(f"  Flights: F1 [0-5], F2 [5-10], F3 [10-15] (sequential)")
    print(f"  Gates: 1 domestic")
    print(f"  Computed NA*: {NA_star_t2}")
    print(f"  Expected NA*: 0 (all can chain on one gate)")
    
    test2_pass = (NA_star_t2 == 0)
    results['test2_sequential_flights'] = {
        'description': 'Three sequential flights with one gate',
        'NA_star': NA_star_t2,
        'expected': 0,
        'passed': test2_pass
    }
    print(f"  TEST 2: {'PASSED' if test2_pass else 'FAILED'}")
    
    # ==========================================================================
    # TEST 3: Maximum clique determines apron need
    # Expected: NA* = max_overlap - gates
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Maximum clique constraint")
    print("=" * 60)
    
    # 3 flights all overlap at time 5, but only 2 gates
    flights_t3 = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 10, 'e': 100, 'f': 100},
        {'id': 'F2', 'terminal': 'D', 'a': 3, 'd': 8, 'e': 100, 'f': 100},
        {'id': 'F3', 'terminal': 'D', 'a': 4, 'd': 12, 'e': 100, 'f': 100},
    ]
    gates_D_t3 = ['D1', 'D2']  # 2 gates for 3 mutually overlapping flights
    
    NA_star_t3 = compute_NA_star(flights_t3, len(gates_D_t3), 'D')
    print(f"  Flights: F1 [0-10], F2 [3-8], F3 [4-12] (all overlap)")
    print(f"  Gates: 2 domestic")
    print(f"  Computed NA*: {NA_star_t3}")
    print(f"  Expected NA*: 1 (3 overlapping - 2 gates)")
    
    test3_pass = (NA_star_t3 == 1)
    results['test3_max_clique'] = {
        'description': 'Three mutually overlapping flights with two gates',
        'NA_star': NA_star_t3,
        'expected': 1,
        'passed': test3_pass
    }
    print(f"  TEST 3: {'PASSED' if test3_pass else 'FAILED'}")
    
    # ==========================================================================
    # TEST 4: Verify full model assigns correct number to apron
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Full model apron assignment")
    print("=" * 60)
    
    flights_t4 = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 5, 'e': 80, 'f': 75},
        {'id': 'F2', 'terminal': 'D', 'a': 2, 'd': 7, 'e': 60, 'f': 50},
        {'id': 'F3', 'terminal': 'I', 'a': 1, 'd': 6, 'e': 120, 'f': 110},
        {'id': 'F4', 'terminal': 'I', 'a': 6, 'd': 10, 'e': 90, 'f': 95},
    ]
    p_matrix_t4 = {('F1', 'F2'): 30, ('F1', 'F3'): 10, ('F3', 'F4'): 40}
    gates_D_t4 = ['D1', 'D2']
    gates_I_t4 = ['I1']
    all_gates_t4 = gates_D_t4 + gates_I_t4 + [apron]
    
    dist_gg_t4 = {(g1, g2): (10 if g1 == g2 else 100) for g1 in all_gates_t4 for g2 in all_gates_t4}
    dist_ge_t4 = {g: (20 if g != apron else 400) for g in all_gates_t4}
    
    model_t4, x_t4, y_t4, NA_star_t4 = build_linear_AGAP_model(
        flights_t4, p_matrix_t4, gates_D_t4, gates_I_t4, apron, dist_gg_t4, dist_ge_t4
    )
    model_t4.optimize()
    
    # Count apron assignments
    apron_count = sum(1 for (fid, g) in x_t4 if g == apron and x_t4[(fid, g)].X > 0.5)
    
    print(f"  Flights: 2 domestic (overlap), 2 international (sequential)")
    print(f"  Gates: 2 domestic, 1 international")
    print(f"  Computed NA*: {NA_star_t4}")
    print(f"  Actual apron assignments: {apron_count}")
    print(f"  Expected NA*: 0 (sufficient gates)")
    
    test4_pass = (NA_star_t4 == apron_count == 0)
    results['test4_full_model'] = {
        'description': 'Full model verification with sufficient gates',
        'NA_star': NA_star_t4,
        'apron_assigned': apron_count,
        'objective': model_t4.ObjVal if model_t4.Status == GRB.OPTIMAL else None,
        'passed': test4_pass
    }
    print(f"  TEST 4: {'PASSED' if test4_pass else 'FAILED'}")
    
    # ==========================================================================
    # TEST 5: Verify no two overlapping flights on same fixed gate
    # ==========================================================================
    print("\n" + "=" * 60)
    print("TEST 5: No overlap on same fixed gate")
    print("=" * 60)
    
    # Use a scenario where apron must be used
    flights_t5 = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 10, 'e': 100, 'f': 100},
        {'id': 'F2', 'terminal': 'D', 'a': 5, 'd': 15, 'e': 100, 'f': 100},
        {'id': 'F3', 'terminal': 'D', 'a': 8, 'd': 20, 'e': 100, 'f': 100},
    ]
    gates_D_t5 = ['D1']
    gates_I_t5 = []
    all_gates_t5 = gates_D_t5 + [apron]
    
    dist_gg_t5 = {(g1, g2): 100 for g1 in all_gates_t5 for g2 in all_gates_t5}
    dist_ge_t5 = {g: 50 for g in all_gates_t5}
    p_matrix_t5 = {}
    
    model_t5, x_t5, _, NA_star_t5 = build_linear_AGAP_model(
        flights_t5, p_matrix_t5, gates_D_t5, gates_I_t5, apron, dist_gg_t5, dist_ge_t5
    )
    model_t5.optimize()
    
    # Check no overlap on D1
    assignments = {fid: g for (fid, g) in x_t5 if x_t5[(fid, g)].X > 0.5}
    overlap_violation = False
    for i, fi in enumerate(flights_t5):
        for fj in flights_t5[i+1:]:
            if assignments.get(fi['id']) == assignments.get(fj['id']) == 'D1':
                if flights_overlap(fi, fj):
                    overlap_violation = True
    
    print(f"  Flights: F1 [0-10], F2 [5-15], F3 [8-20] (all overlap)")
    print(f"  Gates: 1 domestic")
    print(f"  NA*: {NA_star_t5}")
    print(f"  Assignments: {assignments}")
    print(f"  Overlap violation: {overlap_violation}")
    
    test5_pass = not overlap_violation
    results['test5_no_overlap_violation'] = {
        'description': 'Verify no overlapping flights on same fixed gate',
        'assignments': assignments,
        'overlap_violation': overlap_violation,
        'passed': test5_pass
    }
    print(f"  TEST 5: {'PASSED' if test5_pass else 'FAILED'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_passed = all(r['passed'] for r in results.values())
    for name, res in results.items():
        status = "PASSED" if res['passed'] else "FAILED"
        print(f"  {name}: {status}")
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return results


def run_sensitivity_analysis():
    """
    Perform sensitivity analysis by varying key parameters.
    """
    results = {}
    apron = 'APR'
    
    # ==========================================================================
    # BASE SCENARIO
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    
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
    
    # ==========================================================================
    # ANALYSIS 1: Varying Number of Domestic Gates
    # ==========================================================================
    print("\n--- Analysis 1: Varying Domestic Gate Count ---")
    gate_variations = []
    
    # Use a scenario with more overlapping flights to see gate impact
    flights_gate_test = [
        {'id': 'F1', 'terminal': 'D', 'a': 0, 'd': 60, 'e': 150, 'f': 140},
        {'id': 'F2', 'terminal': 'D', 'a': 20, 'd': 80, 'e': 120, 'f': 110},
        {'id': 'F3', 'terminal': 'D', 'a': 40, 'd': 100, 'e': 100, 'f': 95},
        {'id': 'F4', 'terminal': 'D', 'a': 60, 'd': 120, 'e': 80, 'f': 75},
        {'id': 'F5', 'terminal': 'D', 'a': 80, 'd': 140, 'e': 90, 'f': 85},
        {'id': 'F6', 'terminal': 'I', 'a': 0, 'd': 50, 'e': 200, 'f': 180},
        {'id': 'F7', 'terminal': 'I', 'a': 40, 'd': 100, 'e': 160, 'f': 150},
        {'id': 'F8', 'terminal': 'I', 'a': 90, 'd': 150, 'e': 140, 'f': 130},
    ]
    
    p_matrix_gate = {
        ('F1', 'F2'): 25, ('F1', 'F3'): 15,
        ('F2', 'F3'): 20, ('F2', 'F4'): 10,
        ('F3', 'F4'): 25, ('F3', 'F5'): 15,
        ('F4', 'F5'): 20,
        ('F6', 'F7'): 35, ('F6', 'F8'): 20,
        ('F7', 'F8'): 30,
    }
    
    for num_gates in [1, 2, 3, 4]:
        gates_D = [f'D{i+1}' for i in range(num_gates)]
        gates_I = ['I1', 'I2']  # Keep international gates fixed
        all_gates = gates_D + gates_I + [apron]
        
        dist_gg = {(g1, g2): (0 if g1 == g2 else 150) for g1 in all_gates for g2 in all_gates}
        dist_ge = {g: (100 if g != apron else 500) for g in all_gates}
        
        try:
            model, x, _, NA_star = build_linear_AGAP_model(
                flights_gate_test, p_matrix_gate, gates_D, gates_I, apron, dist_gg, dist_ge
            )
            model.optimize()
            
            if model.Status == GRB.OPTIMAL:
                obj = model.ObjVal
                assignments = {fid: g for (fid, g) in x if x[(fid, g)].X > 0.5}
                apron_count = sum(1 for g in assignments.values() if g == apron)
            elif model.Status == GRB.INFEASIBLE:
                # Compute IIS to understand infeasibility
                model.computeIIS()
                infeasible_constrs = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
                print(f"    Infeasible constraints: {infeasible_constrs[:5]}...")  # Show first 5
                obj = None
                apron_count = None
            else:
                obj = None
                apron_count = None
        except Exception as e:
            print(f"    Exception: {e}")
            obj = None
            NA_star = None
            apron_count = None
        
        gate_variations.append({
            'domestic_gates': num_gates,
            'NA_star': NA_star,
            'apron_assigned': apron_count,
            'objective': obj
        })
        if obj:
            print(f"  {num_gates} gates: NA*={NA_star}, Apron={apron_count}, Objective={obj:.2f}")
        else:
            print(f"  {num_gates} gates: Infeasible or error")
    
    results['gate_variation'] = gate_variations
    
    # ==========================================================================
    # ANALYSIS 2: Varying Transfer Passenger Count (scaling p_ij)
    # ==========================================================================
    print("\n--- Analysis 2: Varying Transfer Passenger Multiplier ---")
    transfer_variations = []
    
    gates_D = base_gates_D
    gates_I = base_gates_I
    all_gates = gates_D + gates_I + [apron]
    dist_gg = {(g1, g2): (0 if g1 == g2 else 150) for g1 in all_gates for g2 in all_gates}
    dist_ge = {g: (100 if g != apron else 500) for g in all_gates}
    
    for multiplier in [0.5, 1.0, 1.5, 2.0, 3.0]:
        scaled_p = {k: int(v * multiplier) for k, v in base_p_matrix.items()}
        
        model, x, _, NA_star = build_linear_AGAP_model(
            base_flights, scaled_p, gates_D, gates_I, apron, dist_gg, dist_ge
        )
        model.optimize()
        
        obj = model.ObjVal if model.Status == GRB.OPTIMAL else None
        
        transfer_variations.append({
            'multiplier': multiplier,
            'objective': obj,
            'NA_star': NA_star
        })
        print(f"  Multiplier {multiplier}x: Objective={obj:.2f}" if obj else f"  Multiplier {multiplier}x: Infeasible")
    
    results['transfer_variation'] = transfer_variations
    
    # ==========================================================================
    # ANALYSIS 3: Varying Apron Distance Penalty
    # ==========================================================================
    print("\n--- Analysis 3: Varying Apron Distance Penalty ---")
    apron_dist_variations = []
    
    for apron_dist in [200, 500, 1000, 2000]:
        dist_ge_var = {g: (100 if g != apron else apron_dist) for g in all_gates}
        
        model, x, _, NA_star = build_linear_AGAP_model(
            base_flights, base_p_matrix, gates_D, gates_I, apron, dist_gg, dist_ge_var
        )
        model.optimize()
        
        obj = model.ObjVal if model.Status == GRB.OPTIMAL else None
        
        apron_dist_variations.append({
            'apron_distance': apron_dist,
            'objective': obj,
            'NA_star': NA_star
        })
        print(f"  Apron dist {apron_dist}: Objective={obj:.2f}" if obj else f"  Apron dist {apron_dist}: Infeasible")
    
    results['apron_distance_variation'] = apron_dist_variations
    
    # ==========================================================================
    # ANALYSIS 4: Varying Gate-to-Gate Distance
    # ==========================================================================
    print("\n--- Analysis 4: Varying Gate-to-Gate Distance ---")
    gate_dist_variations = []
    
    for gate_dist in [50, 100, 150, 200, 300]:
        dist_gg_var = {(g1, g2): (0 if g1 == g2 else gate_dist) for g1 in all_gates for g2 in all_gates}
        
        model, x, _, NA_star = build_linear_AGAP_model(
            base_flights, base_p_matrix, gates_D, gates_I, apron, dist_gg_var, dist_ge
        )
        model.optimize()
        
        obj = model.ObjVal if model.Status == GRB.OPTIMAL else None
        
        gate_dist_variations.append({
            'gate_distance': gate_dist,
            'objective': obj,
            'NA_star': NA_star
        })
        print(f"  Gate dist {gate_dist}: Objective={obj:.2f}" if obj else f"  Gate dist {gate_dist}: Infeasible")
    
    results['gate_distance_variation'] = gate_dist_variations
    
    # ==========================================================================
    # ANALYSIS 5: Varying Non-Transfer Passengers (e_i + f_i)
    # ==========================================================================
    print("\n--- Analysis 5: Varying Non-Transfer Passenger Multiplier ---")
    nontransfer_variations = []
    
    for multiplier in [0.5, 1.0, 1.5, 2.0]:
        scaled_flights = []
        for f in base_flights:
            scaled_f = f.copy()
            scaled_f['e'] = int(f['e'] * multiplier)
            scaled_f['f'] = int(f['f'] * multiplier)
            scaled_flights.append(scaled_f)
        
        model, x, _, NA_star = build_linear_AGAP_model(
            scaled_flights, base_p_matrix, gates_D, gates_I, apron, dist_gg, dist_ge
        )
        model.optimize()
        
        obj = model.ObjVal if model.Status == GRB.OPTIMAL else None
        
        nontransfer_variations.append({
            'multiplier': multiplier,
            'objective': obj,
            'NA_star': NA_star
        })
        print(f"  Multiplier {multiplier}x: Objective={obj:.2f}" if obj else f"  Multiplier {multiplier}x: Infeasible")
    
    results['nontransfer_variation'] = nontransfer_variations
    
    return results




if __name__ == "__main__":
    print("=" * 70)
    print("AGAP MODEL VERIFICATION AND SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Run verification
    verification_results = run_verification_tests()
    
    # Run sensitivity analysis
    sensitivity_results = run_sensitivity_analysis()
    



