from basic_pitch.inference import predict
import pretty_midi

def transcribe_audio(audio_path: str) -> pretty_midi.PrettyMIDI:
    """
    Runs Spotify's Basic Pitch on an audio file and returns PrettyMIDI.
    """
    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=0.5,       # Higher = fewer spurious note triggers
        frame_threshold=0.3,       # Confidence threshold for sustained notes
        minimum_note_length=58,    # Minimum note duration in ms (~58ms avoids pick scrape)
        minimum_frequency=80.0,    # ~E2 lowest guitar string in standard tuning
        maximum_frequency=1200.0   # Cutoff above typical guitar range
    )
    return midi_data

def extract_chord_sequence(midi_obj: pretty_midi.PrettyMIDI, onset_tolerance: float = 0.07) -> list[tuple[int, ...]]:
    """
    Extracts chronological pitch events. Simultaneous notes (dyads/chords)
    within `onset_tolerance` seconds are grouped into sorted tuples.
    Example output: [(64,), (60, 67), (62, 69)]
    """
    raw_notes = []
    for inst in midi_obj.instruments:
        if not inst.is_drum:
            raw_notes.extend(inst.notes)

    if not raw_notes:
        return []

    # Sort strictly by onset time, then by pitch
    raw_notes.sort(key=lambda n: (n.start, n.pitch))

    chord_sequence: list[tuple[int, ...]] = []
    current_cluster: list[int] = [raw_notes[0].pitch]
    cluster_anchor_time = raw_notes[0].start

    for note in raw_notes[1:]:
        # If this note starts within the tolerance window of the cluster anchor:
        if note.start - cluster_anchor_time <= onset_tolerance:
            if note.pitch not in current_cluster:
                current_cluster.append(note.pitch)
        else:
            chord_sequence.append(tuple(sorted(current_cluster)))
            current_cluster = [note.pitch]
            cluster_anchor_time = note.start

    if current_cluster:
        chord_sequence.append(tuple(sorted(current_cluster)))

    return chord_sequence

from difflib import SequenceMatcher

def pretty_chord_names(chord: tuple[int, ...]) -> str:
    return "+".join(pretty_midi.note_number_to_name(p) for p in chord)

def evaluate_guitar_riff(target_midi_path: str, student_audio_path: str):
    # 1. Parse target MIDI
    target_midi = pretty_midi.PrettyMIDI(target_midi_path)
    target_seq = extract_chord_sequence(target_midi, onset_tolerance=0.04)

    # 2. Transcribe student audio via Basic Pitch
    student_midi = transcribe_audio(student_audio_path)
    student_seq = extract_chord_sequence(student_midi, onset_tolerance=0.07)

    # 3. Sequence alignment
    matcher = SequenceMatcher(None, target_seq, student_seq)
    accuracy = matcher.ratio() * 100

    print(f"--- Riff Evaluation ---")
    print(f"Overall Accuracy: {accuracy:.1f}%\n")

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        target_chunk = [pretty_chord_names(c) for c in target_seq[i1:i2]]
        student_chunk = [pretty_chord_names(c) for c in student_seq[j1:j2]]

        if tag == 'equal':
            print(f"✅ Correct:      {', '.join(student_chunk)}")
        elif tag == 'replace':
            print(f"❌ Wrong Event:  Played [{', '.join(student_chunk)}], Expected [{', '.join(target_chunk)}]")
        elif tag == 'delete':
            print(f"⚠️  Missed:       Omitted [{', '.join(target_chunk)}]")
        elif tag == 'insert':
            print(f"⚠️  Extra Notes:  Added [{', '.join(student_chunk)}]")