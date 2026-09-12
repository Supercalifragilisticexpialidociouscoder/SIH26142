import sen2sr
import inspect

def main():
    print("=== SEN2SR API Verification ===")
    
    # Let's see what is exposed at the top level
    print(f"Version: {getattr(sen2sr, '__version__', 'Unknown')}")
    
    top_level = dir(sen2sr)
    print(f"Top-level exports: {top_level}")
    
    # We are looking for something like load_model, SEN2SR, or predict
    models = [name for name in top_level if 'model' in name.lower() or 'sen2sr' in name.lower()]
    print(f"Potential model classes/functions: {models}")
    
    # Let's inspect the SEN2SR class if it exists
    if hasattr(sen2sr, 'SEN2SR'):
        model_cls = sen2sr.SEN2SR
        print("\n=== SEN2SR Class ===")
        print(inspect.getdoc(model_cls))
        
        # Check signature of forward or predict
        if hasattr(model_cls, 'forward'):
            sig = inspect.signature(model_cls.forward)
            print(f"\nForward signature: {sig}")
            
    # Also check if there is a 'predict' or 'load' function
    if hasattr(sen2sr, 'predict'):
        sig = inspect.signature(sen2sr.predict)
        print(f"\nPredict signature: {sig}")
        print(inspect.getdoc(sen2sr.predict))
        
if __name__ == "__main__":
    main()
