# PMLDL-Course-Project CV-component

## Setup

1. Setup the environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Insert screenshots in the `screens_for_detecting` folder

4. Run the `objects_detector.py` script to detect objects in the screenshots:
   ```bash
   python objects_detector.py
   ```

The screenshots with detected objects will be saved in the `output` folder.
