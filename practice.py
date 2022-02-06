from pathlib import Path
import numpy as np

info = {
    'name' : ci.name,
    'contexts' : note_contexts,
    'notes' : note_all,
    'spks' : note_spks,
    'onsets' : note_onsets,
    'durations' : note_durations,
    'peth' : peth,
    'peth_ts' : pi.time_bin,
    'peth_note' : peth_note_all
}

save_dir = Path(r'C:\Users\jahn02\Box\PythonToolbox\premotor_spike_decoding\data')
file_name = 'sample_cluster'
save_path = save_dir / file_name
np.save(save_path, info)

