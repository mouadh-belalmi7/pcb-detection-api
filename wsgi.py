<<<<<<< HEAD
import os
import sys

# Add the project directory to the sys.path
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app

if __name__ == "__main__":
=======
import os
import sys

# Add the project directory to the sys.path
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app

if __name__ == "__main__":
>>>>>>> 518cca968ba8bfdd0ffde9866e726a735025351d
    app.run()