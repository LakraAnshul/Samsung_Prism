import sys
from pathlib import Path

# Add the project root to sys.path so we can import from scripts
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retrieve_text import retrieve_text

def test_empty_query():
    try:
        retrieve_text("")
        print("Failed: Empty query should exit")
        return False
    except SystemExit:
        pass
    return True

def test_whitespace_query():
    try:
        retrieve_text("     ")
        print("Failed: Whitespace query should exit")
        return False
    except SystemExit:
        pass
    return True

def test_normal_query():
    res = retrieve_text("washer isn't filling with water", model="WA5471ABP/XAA")
    assert res["query"] == "washer isn't filling with water"
    assert len(res["results"]) > 0
    return True

def test_exact_technical_term():
    res = retrieve_text("debris filter", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_error_like_query():
    res = retrieve_text("washer won't start", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_model_specific_query():
    res = retrieve_text("WA5471ABP isn't draining", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_safety_language():
    res = retrieve_text("how do I inspect the power cord safely", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_short_query():
    res = retrieve_text("draining", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def test_mixed_case():
    res1 = retrieve_text("DEBRIS FILTER", model="WA5471ABP/XAA")
    res2 = retrieve_text("debris filter", model="WA5471ABP/XAA")
    assert len(res1["results"]) > 0
    assert len(res2["results"]) > 0
    # Both queries should retrieve overlapping relevant chunks
    chunks1 = set(c["chunk_id"] for c in res1["results"])
    chunks2 = set(c["chunk_id"] for c in res2["results"])
    assert len(chunks1 & chunks2) > 0
    return True

def test_alphanumeric_identifier():
    res = retrieve_text("WA5471ABP", model="WA5471ABP/XAA")
    assert len(res["results"]) > 0
    return True

def main():
    tests = [
        test_empty_query,
        test_whitespace_query,
        test_normal_query,
        test_exact_technical_term,
        test_error_like_query,
        test_model_specific_query,
        test_safety_language,
        test_short_query,
        test_mixed_case,
        test_alphanumeric_identifier
    ]
    
    passed = 0
    for t in tests:
        try:
            if t():
                passed += 1
                print(f"{t.__name__} passed")
        except AssertionError:
            print(f"{t.__name__} FAILED (assertion)")
        except Exception as e:
            print(f"{t.__name__} FAILED ({e})")
            
    print(f"\nTests: {passed}/{len(tests)} passed")
    if passed != len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()
