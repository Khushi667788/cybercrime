from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.utils import print_separator

def main():
    print("Loading data...")
    df = load_data("data/cybercrime_data.csv")
    print_separator()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print_separator()
    
    print("Training model...")
    model = train_model(X_train, y_train)
    print_separator()
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    print_separator()

if __name__ == "__main__":
    main()
