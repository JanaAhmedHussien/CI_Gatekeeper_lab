# app.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import os

def main():
    print("Generating synthetic data (Lightning fast)...")
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    # Run fixed one
    print("Training 'Expensive' Model...")
    model = RandomForestClassifier(n_estimators=10)
    model.fit(Xy, y)
    
    # Check for a specific environment variable to simulate a failure if needed
    if os.getenv("SIMULATE_FAILURE") == "true":
        raise RuntimeError("Simulated GPU Memory Overflow!")

    print(f"Model trained successfully. Features processed: {model.n_features_in_}")

if __name__ == "__main__":
    try:
        # Create the log file pre-emptively for the artifact step
        with open("error_logs.txt", "w") as f:
            f.write("Log initialized. Starting training sequence...")
        main()
    except Exception as e:
        with open("error_logs.txt", "a") as f:
            f.write(f"\nFATAL ERROR: {str(e)}")
        exit(1) # Ensure the job fails so GHA catches it
