import railroad._bindings as b

def main():
    print("Testing if bindings are exposed...")
    if hasattr(b, "get_relaxed_expected_costs"):
        print("✅ get_relaxed_expected_costs is available.")
    else:
        print("❌ get_relaxed_expected_costs is NOT available.")

    if hasattr(b, "get_achievers_for_fluent"):
        print("✅ get_achievers_for_fluent is available.")
    else:
        print("❌ get_achievers_for_fluent is NOT available.")

if __name__ == "__main__":
    main()
