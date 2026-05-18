import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.predict_may18 import May18Predictor

if __name__ == '__main__':
    May18Predictor().run()
