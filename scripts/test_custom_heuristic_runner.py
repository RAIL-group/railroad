import sys
import os

# Ensure the packages are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../packages/railroad/src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../packages/railroad/tests")))

from test_custom_heuristic import test_custom_heuristic_integration, test_sandbox_evaluator_tournament

if __name__ == "__main__":
    print("Running test_custom_heuristic_integration...")
    try:
        test_custom_heuristic_integration()
        print("✅ Custom heuristic integration test PASSED!")
    except Exception as e:
        print("❌ Test failed with exception:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nRunning test_sandbox_evaluator_tournament...")
    try:
        test_sandbox_evaluator_tournament()
        print("✅ Sandbox evaluator tournament test PASSED!")
    except Exception as e:
        print("❌ Tournament test failed with exception:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
