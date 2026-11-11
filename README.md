# dual_sdm_unips

sdm_unips loader and student-model code from [abaowannasleep/dual-sdm](https://github.com/abaowannasleep/dual-sdm) commit `0fe18076b44644df1ada9b2f860449b71577cb16`. This repository has its own commit and does not include that source history.

The loader is `modules/io/multicolorps.py`. `modules/model/student_model.py` wraps `modules/model/model_fast.py`. The other modules are the local files those three import.

Run Python from this repository root:

```python
from modules.io.multicolorps import MultiColorPS
from modules.model.student_model import StudentModel
```

The source packages ship without `__init__.py`. These imports work with this root on `PYTHONPATH`.
