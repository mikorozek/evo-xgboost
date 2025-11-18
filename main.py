import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score

from src.evo_xgboost.data import load_and_split_data
from src.evo_xgboost.evolution import run_evolution
from src.evo_xgboost.settings import DATA_URL, PARAM_RANGES

def main():
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_URL)
    print("Data loaded and preprocessed.")

    best_individual = run_evolution(X_train, y_train, X_test, y_test)

    print("\nVerifying final model on test set...")
    final_params = best_individual.copy()
    for param, value in final_params.items():
        if PARAM_RANGES[param][0] == 'discrete':
            final_params[param] = int(value)

    final_model = xgb.XGBClassifier(
        **final_params,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    final_model.fit(X_train, y_train, verbose=False)
    y_pred = final_model.predict(X_test)
    final_f1 = f1_score(y_test, y_pred)
    final_accuracy = accuracy_score(y_test, y_pred)

    print(f"Final F1-score on test set: {final_f1:.4f}")
    print(f"Final Accuracy on test set: {final_accuracy:.4f}")

if __name__ == '__main__':
    main()
