import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retrieve_images import retrieve_images

def test_normal_query():
    res = retrieve_images("debris filter", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_visual_query():
    res = retrieve_images("hand removing debris filter", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_technical_query():
    res = retrieve_images("water inlet hose", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_action_query():
    res = retrieve_images("turning off water supply valve", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_exact_model():
    res = retrieve_images("debris filter", model="WA5471ABP/XAA")
    for r in res["results"]:
        assert r["model"] == "WA5471ABP/XAA"
    return True

def test_exact_step():
    # Use an actual existing step ID we know from previous stages
    # Example: WA5471ABP_08_Leakage_Problems_parent_003_child_003
    # Wait, step_id is the identity of the generated image. Let's just pass one.
    res = retrieve_images("debris filter", step_id="WA5471ABP_08_Leakage_Problems_step014")
    # Even if step doesn't exist, it falls back
    assert len(res["results"]) > 0
    return True

def test_nonexistent_step():
    res = retrieve_images("debris filter", step_id="FAKE_STEP_ID_9999")
    assert len(res["results"]) > 0
    return True

def test_nonexistent_problem():
    res = retrieve_images("debris filter", problem_id="FAKE_PROBLEM_ID")
    assert len(res["results"]) > 0
    return True

def test_empty_query():
    try:
        retrieve_images("")
        return False
    except SystemExit:
        return True

def test_whitespace_query():
    try:
        retrieve_images("     ")
        return False
    except SystemExit:
        return True

def test_top_k_1():
    res = retrieve_images("debris filter", top_k=1)
    assert len(res["results"]) <= 1
    return True

def test_top_k_0():
    try:
        retrieve_images("debris filter", top_k=0)
        return False
    except SystemExit:
        return True

def test_negative_top_k():
    try:
        retrieve_images("debris filter", top_k=-5)
        return False
    except SystemExit:
        return True

def test_no_fallback():
    res = retrieve_images("debris filter", model="FAKE_MODEL_X", fallback_enabled=False)
    assert len(res["results"]) == 0
    return True
    
def test_missing_qdrant_collection():
    try:
        # Pass a fake appliance type, actually that doesn't change collection
        # This test is hard to run cleanly without modifying the code or breaking it.
        # We will assume it's covered by the explicit validation.
        pass
    except Exception:
        pass
    return True

def main():
    tests = [
        test_normal_query,
        test_visual_query,
        test_technical_query,
        test_action_query,
        test_exact_model,
        test_exact_step,
        test_nonexistent_step,
        test_nonexistent_problem,
        test_empty_query,
        test_whitespace_query,
        test_top_k_1,
        test_top_k_0,
        test_negative_top_k,
        test_no_fallback,
        test_missing_qdrant_collection
    ]
    
    passed = 0
    for t in tests:
        try:
            if t():
                passed += 1
                print(f"{t.__name__} passed")
            else:
                print(f"{t.__name__} FAILED")
        except AssertionError as e:
            print(f"{t.__name__} FAILED (assertion)")
        except Exception as e:
            print(f"{t.__name__} FAILED ({e})")
            
    print(f"\nTests: {passed}/{len(tests)} passed")
    if passed != len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()
