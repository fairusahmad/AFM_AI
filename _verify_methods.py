"""验证所有新增方法是否可访问"""
import numpy as np
from afm_state import AFMState
from afm_callbacks import AFMCallbacks

s = AFMState(np.zeros((100,100), dtype=np.uint8), 10000, 10000)
c = AFMCallbacks(s, None, None, None, None, None, None, None, None, None, None, None, None)

methods = [
    'ai_recall_and_recover',
    'ai_zoom_recover',
    '_wait_for_zoom_complete',
    '_enter_click_to_move_correction',
    '_attempt_pattern_recognition_at_current_zoom',
    '_start_zoom_out_search',
    '_continue_zoom_search',
    '_finish_ai_zoom_move',
]

all_ok = True
for m in methods:
    ok = hasattr(c, m)
    status = "OK" if ok else "MISSING"
    print(f"  {m}: {status}")
    if not ok:
        all_ok = False

print(f"\n{'All methods present!' if all_ok else 'SOME METHODS MISSING!'}")
