import os
import sys

# Ensure the parent directory is in the path so we can import hospital_api
# without installation issues.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hospital_api import app, init_environment

def main():
    # Initialize with default settings as done in hospital_api.py
    init_environment()
    
    # OpenEnv default configurations
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 7860))
    is_prod = os.environ.get('ENV') == 'production'
    
    app.run(debug=not is_prod, host=host, port=port)

if __name__ == '__main__':
    main()
