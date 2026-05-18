import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ensemble_q1 import Q1Ensemble

if __name__ == '__main__':
    Q1Ensemble().run()
