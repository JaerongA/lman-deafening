"""
Syllable network analysis and calculates transition entropy
"""

from pyfinch.analysis import SongInfo
from pyfinch.database.load import ProjectLoader, DBInfo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from util import save
import warnings
warnings.filterwarnings('ignore')

def nb_song_note_in_bout(song_notes: str, bout: str) -> int:
    """Returns the number of song notes within a bout"""
    nb_song_note_in_bout = len([note for note in song_notes if note in bout])
    return nb_song_note_in_bout


def get_syl_color(bird_id: str) -> dict:
    """Map colors to each syllable"""
    from pyfinch.analysis import sequence_color
    import copy

    # Load database
    db = ProjectLoader().load_db()
    df = db.to_dataframe(f"""SELECT (introNotes || songNote || calls || '*') AS note_sequence, 
                        introNotes, songNote, calls FROM bird WHERE birdID='{bird_id}'""")

    note_seq = df['note_sequence'][0]
    intro_notes = df['introNotes'][0]
    song_notes = df['songNote'][0]
    calls = df['calls'][0]

    syl_color = dict()
    sequence_color2 = copy.deepcopy(sequence_color)

    for i, note in enumerate(note_seq[:-1]):
        if note in song_notes:
            syl_color[note] = sequence_color2['song_note'].pop(0)
        elif note in intro_notes:
            syl_color[note] = sequence_color2['intro'].pop(0)
        elif note in calls:
            syl_color[note] = sequence_color2['call'].pop(0)
        else:
            syl_color[note] = sequence_color2['intro'].pop(0)
    syl_color['*'] = 'y'  # syllable stop

    return note_seq, syl_color


def get_trans_matrix(syllables: str, note_seq: str, normalize=False) -> np.ndarray:
    """Build a syllable transition matrix"""

    trans_matrix = np.zeros((len(note_seq), len(note_seq)), dtype='int16')  # initialize the matrix
    # print(syllables)
    for i, note in enumerate(syllables):
        if i < len(syllables) - 1:
            if not (syllables[i] in note_seq) or not (syllables[i + 1] in note_seq):
                continue
            # print(syllables[i] + '->' + syllables[i + 1])  # for debugging
            ind1 = note_seq.index(syllables[i])
            ind2 = note_seq.index(syllables[i + 1])
            if ind1 < len(note_seq) - 1:
                trans_matrix[ind1, ind2] += 1
    if normalize:
        trans_matrix = trans_matrix / trans_matrix.sum()
    return trans_matrix


def plot_transition_diag(ax, note_seq, syl_network, syl_color,
                         syl_circ_size=450,
                         line_width=0.8,
                         dev_factor=0.6
                         ):
    """Plot syllable transition diagram"""
    import math
    np.random.seed(0)

    # Set node location
    theta = np.linspace(-math.pi, math.pi, num=len(note_seq) + 1)  # for each node

    node_xpos = [math.cos(node) for node in theta]
    node_ypos = [math.sin(node) for node in theta][::-1]

    # Plot the syllable node
    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    ax.scatter(node_xpos[:-1], node_ypos[:-1], s=syl_circ_size, facecolors='w',
               edgecolors=list(syl_color.values()),
               zorder=2.5,
               linewidth=2.5)
    ax.set_xlim([-1.2, 1.2]), ax.set_ylim([-1.2, 1.2])

    circle_size = 0.25  # circle size for the repeat syllable

    for i, (start_node, end_node, weight) in enumerate(syl_network):

        if start_node != end_node:  # transition syllable

            start_nodex = node_xpos[start_node] + (np.random.uniform(-dev_factor, dev_factor, weight) / 10)
            start_nodey = node_ypos[start_node] + (np.random.uniform(-dev_factor, dev_factor, weight) / 10)

            end_nodex = node_xpos[end_node] + (np.random.uniform(-dev_factor, dev_factor, weight) / 10)
            end_nodey = node_ypos[end_node] + (np.random.uniform(-dev_factor, dev_factor, weight) / 10)

            ax.scatter(start_nodex, start_nodey, s=0, facecolors='k')
            ax.scatter(end_nodex, end_nodey, s=0, facecolors='k')
            ax.plot([start_nodex, end_nodex], [start_nodey, end_nodey], 'k',
                    color=list(syl_color.values())[start_node],
                    linewidth=line_width)

        else:  # repeating syllables
            factor = 1.25  # adjust center of the circle for the repeat
            syl_loc = ((np.array(node_xpos) * factor).tolist(), (np.array(node_ypos) * factor).tolist())

            start_nodex = syl_loc[0][start_node] + (np.random.uniform(-dev_factor, dev_factor, weight) / 8)
            start_nodey = syl_loc[1][start_node] + (np.random.uniform(-dev_factor, dev_factor, weight) / 8)

            for x, y in zip(start_nodex, start_nodey):
                circle = plt.Circle((x, y), circle_size, color=list(syl_color.values())[start_node], fill=False,
                                    clip_on=False,
                                    linewidth=0.3)
                ax.add_artist(circle)

        # Set text labeling location
        factor = 1.7
        text_loc = ((np.array(node_xpos) * factor).tolist(), (np.array(node_ypos) * factor).tolist())

        for ind, note in enumerate(note_seq):
            ax.text(text_loc[0][ind], text_loc[1][ind], note_seq[ind], fontsize=15)


def get_syllable_network(trans_matrix: np.ndarray) -> list:
    """
    Build sparse representation of a syllable network

    Parameters
    ----------
    trans_matrix : np.ndarray
        transition matrix
    Returns
    -------
    syl_network : list of tuple (start node, end node, weight)
    """

    start_node = np.transpose(np.nonzero(trans_matrix))[:, 0].T.tolist()
    end_node = np.transpose(np.nonzero(trans_matrix))[:, 1].T.tolist()
    weight = []
    for ind in range(0, len(start_node)):
        weight.append(int(trans_matrix[start_node[ind], end_node[ind]]))

    syl_network = list(zip(start_node, end_node, weight))
    return syl_network


def get_trans_entropy(trans_matrix: np.ndarray, weighted=False, ignore_repeats=False) -> float:
    """
    Calculate transition entropy
    entropy will be equal to zero if all notes transition to only one syllable
    """
    matrix = trans_matrix.copy()

    if ignore_repeats:
        np.fill_diagonal(matrix, 0)

    trans_entropy = []
    for row in matrix:
        if np.sum(row):
            prob = row / np.sum(row)
            entropy = - np.nansum(prob * np.log2(prob))

            if weighted:
                note_prob = np.sum(row) / np.sum(matrix)
                entropy *= note_prob

            trans_entropy.append(entropy)
    # print(trans_entropy)
    trans_entropy = np.mean(trans_entropy)
    return trans_entropy


def get_sequence_linearity(note_seq: str, syl_network: list) -> float:

    nb_unique_transitions = len(syl_network)
    # print(nb_unique_transitions)
    nb_unique_syllables = len(note_seq) - 1  # stop syllable (*) not counted here
    sequence_linearity = nb_unique_syllables / nb_unique_transitions
    # print(nb_unique_syllables)
    return sequence_linearity


def get_sequence_consistency(note_seq: str, trans_matrix: np.ndarray) -> float:

    typical_transition = []
    for i, row in enumerate(trans_matrix):
        max_ind = np.where(row == np.amax(row))
        if ((max_ind[0].shape[0]) == 1) \
                and (
                np.sum(row)):  # skip if there are more than two max weight values or the sum of weights equals zero
            # print(f"{note_seq[i]} -> {note_seq[max_ind[0][0]]}") # starting syllable -> syllable with the highest prob of transition"
            typical_transition.append((note_seq[i], note_seq[max_ind[0][0]]))

    nb_typical_transition = len(typical_transition)
    nb_total_transition = np.count_nonzero(trans_matrix)
    sequence_consistency = nb_typical_transition / nb_total_transition
    return sequence_consistency


def get_song_stereotypy(sequence_linearity: float, sequence_consistency: float) -> float:
    """Average between linearity and consistency"""
    song_stereotypy = (sequence_linearity + sequence_consistency) / 2
    return song_stereotypy


def get_canonical_trans_prob(trans_matrix: np.ndarray, note_seq: set, song_notes: str) -> float:
    """Calculate transition probability within a canonical song motif"""

    # Find number of transitions within a song motif up to the penultimate syllable
    trans_dict = {}
    note_to_use = song_notes[:-1]

    for ind, note in enumerate(note_seq):
        if note in note_to_use:
            trans_dict[note] = trans_matrix[ind]

    # Calculate canonical transition probability
    all_transitions = 0
    canonical_transitions = 0

    for note, transition in trans_dict.items():
        canonical_transitions += transition[note_seq.index(note) + 1]
        all_transitions += np.sum(transition)

    return canonical_transitions / all_transitions


def main():
    """Main function"""

    # Parameters
    nb_row = 10  # for plotting
    font_size = 12

    # Load database
    db = ProjectLoader().load_db()
    db.execute(query)

    # Make database
    if update_db:
        with open('../database/create_song_sequence.sql', 'r') as sql_file:
            db.conn.executescript(sql_file.read())

    # Loop through db
    for row in db.cur.fetchall():

        # Load song info from db
        song_db = DBInfo(row)
        name, path = song_db.load_song_db()

        si = SongInfo(path, name, update=update)  # song object

        # Get syllable color
        note_seq, syl_color = get_syl_color(song_db.birdID)

        # Store results per context
        song_bouts = dict()
        nb_bouts = dict()  # nb of song bouts per context
        trans_entropy = dict()
        weighted_entropy = dict()
        sequence_linearity = dict()
        sequence_consistency = dict()
        song_stereotypy = dict()
        canonical_trans_prob = dict()

        for i, context in enumerate(sorted(set(si.contexts), reverse=True)):
            bout_list = []
            syllable_list = [syllable for syllable, _context in zip(si.syllables, si.contexts) if
                             _context == context]
            for syllables in syllable_list:
                bout = [bout for bout in syllables.split('*') if nb_song_note_in_bout(song_db.songNote, bout)]
                if bout:
                    bout_list.append(bout[0])
            if bout_list:
                song_bouts[context] = '*'.join(bout_list) + '*'
                nb_bouts[context] = len(song_bouts[context].split('*')[:-1])

        for i, context in enumerate(song_bouts.keys()):

            # remove a note from note sequence if it doesn't appear in a session
            intersection_ = set(note_seq).intersection(set(song_bouts[context]))
            for note in note_seq:
                if note not in intersection_:
                    note_seq = note_seq.replace(note, '')

            # Get transition matrix
            trans_matrix = []
            trans_matrix = get_trans_matrix(song_bouts[context], note_seq)

            # Plot transition matrix
            if not 'fig' in locals():
                fig = plt.figure(figsize=(len(song_bouts.keys()) * 4, 10), dpi=350)
                plt.suptitle(si.name, y=.98, fontsize=font_size)

            ax = plt.subplot2grid((nb_row, len(song_bouts.keys())), (0, i), rowspan=3, colspan=1)
            y_max = trans_matrix.max() + (10 - trans_matrix.max() % 10)
            ax = sns.heatmap(trans_matrix, cmap=cmap, annot=True, vmin=0, vmax=y_max,
                             linewidth=0.2, linecolor='k',
                             cbar_kws=dict(ticks=[0, y_max / 2, y_max], label='# of transitions'),
                             annot_kws={"fontsize": 6}, fmt='g')
            ax.set_title(f"{context}", y=2)
            ax.tick_params(axis='both', which='major', labelsize=10,
                           labelbottom=False, bottom=False, top=False,
                           labeltop=True)  # move the x-tick label on the top of the figure
            ax.set_aspect(aspect=1)
            ax.set_xticks(np.arange(0, len(note_seq)) + 0.5)
            ax.set_yticks(np.arange(0, len(note_seq)) + 0.5)
            ax.set_xticklabels(note_seq, weight='bold', fontsize=12)
            ax.set_yticklabels(note_seq, rotation=0, weight='bold', fontsize=12)
            ax.tick_params(axis=u'both', which=u'both', length=0)
            ax.set_xlim(ax.get_xlim()[0] - 0.01, ax.get_xlim()[1] + 0.01)
            ax.set_ylim(ax.get_ylim()[0] + 0.01, ax.get_ylim()[1] - 0.01)

            # Calculate transition entropy
            # entropy = 0 when there's a single transition and increases with more branching points
            trans_entropy[context] = get_trans_entropy(trans_matrix, ignore_repeats=ignore_repeats)
            weighted_entropy[context] = get_trans_entropy(trans_matrix, weighted=True, ignore_repeats=ignore_repeats)

            # Build the syllable network (start & end nodes and weights)
            syl_network = get_syllable_network(trans_matrix)

            # Calculate transition metrics
            sequence_linearity[context] = get_sequence_linearity(note_seq, syl_network)

            sequence_consistency[context] = get_sequence_consistency(note_seq, trans_matrix)

            song_stereotypy[context] = get_song_stereotypy(sequence_linearity[context],
                                                           sequence_consistency[context])

            canonical_trans_prob[context] = get_canonical_trans_prob(trans_matrix, note_seq, song_db.songNote)

            # Plot transition diagram
            ax = plt.subplot2grid((nb_row, len(set(song_bouts.keys()))), (4, i), rowspan=3, colspan=1)
            plot_transition_diag(ax, note_seq, syl_network, syl_color)
            ax_txt = plt.subplot2grid((nb_row, len(set(song_bouts.keys()))), (9, i), rowspan=1, colspan=1)
            ax_txt.axis('off')

            txt_xloc = 0.1
            txt_yloc = -0.4
            txt_inc = 0.15

            ax_txt.text(txt_xloc, txt_yloc, f"""nb bouts = {nb_bouts[context]}""",
                        fontsize=font_size, transform=ax.transAxes)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"""transition entropy = {trans_entropy[context]: .3f}""",
                        fontsize=font_size, transform=ax.transAxes)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"""sequence linearity = {sequence_linearity[context]: .3f}""",
                        fontsize=font_size, transform=ax.transAxes)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"""sequence consistency = {sequence_consistency[context]: .3f}""",
                        fontsize=font_size, transform=ax.transAxes)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"""song stereotypy = {song_stereotypy[context]: .3f}""",
                        fontsize=font_size, transform=ax.transAxes)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"""weighted entropy = {weighted_entropy[context]: .3f}""",
                        fontsize=font_size, transform=ax.transAxes)
            txt_yloc -= txt_inc
            ax_txt.text(txt_xloc, txt_yloc, f"""canonical trans prob = {canonical_trans_prob[context]: .3f}""",
                        fontsize=font_size, transform=ax.transAxes)

        # plt.tight_layout()

        # Save results
        if save_fig:
            save_path = save.make_dir(ProjectLoader().path / 'Analysis', save_folder_name, add_date=True)
            save.save_fig(fig, save_path, si.name, fig_ext=fig_ext, view_folder=view_folder)
        else:
            plt.show()
        del fig

        # Update database
        if update_db:
            for context, val in nb_bouts.items():
                column = 'nbBoutsUndir' if context == 'U' else 'nbBoutsDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val} WHERE songID= {song_db.id}""")

            for context, val in trans_entropy.items():
                column = 'transEntUndir' if context == 'U' else 'transEntDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val: 0.3f} WHERE songID= {song_db.id}""")

            for context, val in sequence_linearity.items():
                column = 'seqLinearityUndir' if context == 'U' else 'seqLinearityDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val: 0.3f} WHERE songID= {song_db.id}""")

            for context, val in sequence_consistency.items():
                column = 'seqConsistencyUndir' if context == 'U' else 'seqConsistencyDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val: 0.3f} WHERE songID= {song_db.id}""")

            for context, val in song_stereotypy.items():
                column = 'songStereotypyUndir' if context == 'U' else 'songStereotypyDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val: 0.3f} WHERE songID= {song_db.id}""")
            db.conn.commit()

            db.create_col('song_sequence', 'WeightedEntropyUndir', 'REAL')
            db.create_col('song_sequence', 'WeightedEntropyDir', 'REAL')
            db.create_col('song_sequence', 'CanonicalTransProbUndir', 'REAL')
            db.create_col('song_sequence', 'CanonicalTransProbDir', 'REAL')
            db.conn.commit()

            for context, val in weighted_entropy.items():
                column = 'WeightedEntropyUndir' if context == 'U' else 'WeightedEntropyDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val: 0.3f} WHERE songID= {song_db.id}""")

            for context, val in canonical_trans_prob.items():
                column = 'CanonicalTransProbUndir' if context == 'U' else 'CanonicalTransProbDir'
                db.cur.execute(f"""UPDATE song_sequence SET {column}={val: 0.3f} WHERE songID= {song_db.id}""")
            db.conn.commit()

    # Convert db to csv
    if update_db:
        db.to_csv('song_sequence')
        print('Done!')


if __name__ == '__main__':

    # Parameters
    update = False  # update/create cache file
    update_db = False
    save_fig = True
    view_folder = True  # open up the figure folder after saving
    fig_ext = '.pdf'
    save_folder_name = 'SequenceAnalysis'
    cmap = "gist_heat_r"  # heatmap color scheme
    ignore_repeats = True  # ignore repeats when calculating entropy

    # SQL statement
    query = "SELECT * FROM song WHERE id=48"
    # query = "SELECT * FROM song"

    main()
