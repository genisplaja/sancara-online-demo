import io
import os
import pickle

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from flask import Flask, render_template, Response, send_from_directory
from werkzeug.security import safe_join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import song_chooser
from utils import listen_pattern, pitch_seq_to_cents, myround, pitch_to_cents

#data = pickle.load(open("test.pkl", "rb"))
data = song_chooser.SongChooser.get_data()

if "SENTRY_DSN" in os.environ:
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=1.0
    )


app = Flask(__name__)

@app.route('/')
def index():
    recordings = list(data.keys())
    recordings.remove('audio')
    return render_template("index.html", recordings=recordings)

@app.route('/recording/<string:title>')
def recording(title):
    if title not in data:
        return "not found", 404
    recording = data[title]

    patterns = recording['parsed_patterns']
    if recording['annotations'] is not None:
        groups = recording['groups_with_annotations']
    else:
        groups = recording['groups']

    return render_template("recording.html", title=title, patterns=patterns.keys(), groups=groups.keys())


@app.route('/recording/<string:title>/<string:pat_type>/<string:identifier>')
def pattern(title, pat_type, identifier):
    ret, code = validate_args(title, pat_type, identifier)
    if ret is not None:
        return ret, code

    recording = data[title]
    has_annotations = False
    if recording['annotations'] is not None:
        groups = recording['groups_with_annotations']
        has_annotations = True
    else:
        groups = recording['groups']
    patterns = recording['parsed_patterns']
    pattern_names = [str(p) for p in patterns.keys()]
    group_names = [str(g) for g in groups.keys()]

    if pat_type == "Group":
        # If the type is "Group" then the identifier is an int index. validate_args has already
        # checked that it's a number and is within bounds
        results = groups[int(identifier)]
    elif pat_type == "Pattern":
        # If type is "Pattern" then the identifier is a dict index into `patterns`. validate_args has
        # checked that the index is valid
        results = patterns[identifier]

    return render_template(
        "recording.html", title=title, patterns=pattern_names, groups=group_names,
        pat_type=pat_type, has_annotations=has_annotations, identifier=identifier, results=results
    )

@app.route('/image/<string:title>/<string:pat_type>/<string:identifier>/<string:part>.png')
def image(title, pat_type, identifier, part):

    ret, code = validate_args(title, pat_type, identifier)
    if ret is not None:
        return ret, code

    output = io.BytesIO()

    recording = data[title]
    annotations = recording['annotations']
    freqs = recording['pitch_freqs']
    times = recording['pitch_times']
    pitch_hop = recording['pitch_hop']
    plot_kwargs = recording['plot_kwargs']
    patterns = recording['parsed_patterns']
    if recording['annotations'] is not None:
        groups = recording['groups_with_annotations']
    else:
        groups = recording['groups']

    if pat_type == "Group":
        # If the type is "Group" then the identifier is an int index. validate_args has already
        # checked that it's a number and is within bounds
        results = groups[int(identifier)]
    elif pat_type == "Pattern":
        # If type is "Pattern" then the identifier is a dict index into `patterns`. validate_args has
        # checked that the index is valid
        results = patterns[identifier]
    
    part = int(part)
    if part >= len(results):
        return "part too long", 400

    fig = plot_patterns(recording, results[part], pat_type, freqs, times, pitch_hop, annotations, plot_kwargs)
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/audio/<string:rec>/<string:fold>/<string:gr>/<string:oc>.wav')
def audio(rec, fold, gr, oc):
    audio_path = safe_join('data', rec, fold, str(gr)+'_'+str(oc)+'.wav')
    audio_dir = os.path.dirname(audio_path)
    audio_fname = os.path.basename(audio_path)
    if not os.path.exists(audio_path):
        return "no such audio file", 404

    return send_from_directory(audio_dir, audio_fname)


def validate_args(title, type, identifier):
    if title not in data:
        return "title not found", 404

    if type not in ['Group', 'Pattern']:
        return f"unknown type ({type})", 400

    recording = data[title]
    if recording['annotations'] is not None:
        groups = recording['groups_with_annotations']
    else:
        groups = recording['groups']

    if type == "Group":
        try:
            identifier = int(identifier)
        except ValueError:
            return "invalid group (not number)", 400
        if identifier >= len(groups):
            return "invalid group", 400

    patterns = recording['parsed_patterns']

    if type == "Pattern" and identifier not in patterns:
        return "invalid pattern", 400

    return None, None



def plot_patterns(rec, pattern_info, option, freqs, times, pitch_hop, annotations, plot_kwargs):
    sp = pattern_info[0]
    l = pattern_info[1]
    if annotations is not None:
        pat = pattern_info[2]
        pat_full = pattern_info[3]
    else:
        pat = None
        pat_full = None
    this_pitch = freqs[int(max(sp-l,0)):int(sp+2*l)]
    this_times = times[int(max(sp-l,0)):int(sp+2*l)]
    mask = np.full((len(this_pitch),), False)

    tonic = plot_kwargs['tonic']
    p1 = pitch_seq_to_cents(this_pitch, tonic)
    figsize = plot_kwargs['figsize']
    cents = plot_kwargs['cents']
    s_len = len(this_pitch)
    pitch_masked = np.ma.masked_where(mask, p1)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    # hide toolbar and title
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.patch.set_facecolor('white')

    fig.subplots_adjust(wspace=0.3, hspace=0.20, top=0.85, bottom=0.1)
    xlabel = 'Time (s)'
    ylabel = f'Cents Above Tonic of {round(tonic)}Hz' if cents else 'Pitch (Hz)'

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    xmin = myround(min(this_times[:s_len]), 5)
    xmax = max(this_times[:s_len])

    sample = pitch_masked.data[:s_len]
    if not set(sample) == {None}:
        ymin_ = min([x for x in sample if x is not None])
        ymin = myround(ymin_, 50)
        ymax = max([x for x in sample if x is not None])
    else:
        ymin=0
        ymax=1000
    
    for s in plot_kwargs['emphasize']:
        assert plot_kwargs['yticks_dict'], \
            "Empasize is for highlighting certain ticks in <yticks_dict>"
        if s in plot_kwargs['yticks_dict']:
            if cents:
                p_ = pitch_to_cents(plot_kwargs['yticks_dict'][s], plot_kwargs['tonic'])
            else:
                p_ = plot_kwargs['yticks_dict'][s]
            ax.axhline(p_, color='#db1f1f', linestyle='--', linewidth=1)

    times_samp = this_times[:s_len]
    pitch_masked_samp = pitch_masked[:s_len]

    times_samp = times_samp[:min(len(times_samp), len(pitch_masked_samp))]
    pitch_masked_samp = pitch_masked_samp[:min(len(times_samp), len(pitch_masked_samp))]
    ax.plot(times_samp, pitch_masked_samp, linewidth=0.7)

    if plot_kwargs['yticks_dict']:
        tick_names = list(plot_kwargs['yticks_dict'].keys())
        tick_loc = [pitch_to_cents(p, plot_kwargs['tonic']) if cents else p \
                    for p in plot_kwargs['yticks_dict'].values()]
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(tick_names)
    
    ax.set_xticks(np.arange(xmin, xmax+1, 1))

    plt.xticks(fontsize=8.5)
    ax.set_facecolor('#f2f2f2')

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((xmin, xmax))
        
    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l,sp)):int(l+min(l,sp))]
    y = y_d[int(min(l,sp)):int(l+min(l,sp))]
    
    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    rect = Rectangle((x_d[int(min(l,sp))], min_y), l*pitch_hop, max_y-min_y, facecolor='lightgrey')
    ax.add_patch(rect)
    
    ax.plot(x, y, linewidth=0.7, color='darkorange')
    ax.axvline(x=x_d[int(min(l,sp))], linestyle="dashed", color='black', linewidth=0.8)

    fig.canvas.draw()
    fig.canvas.flush_events()
    if 'Group' in option:
        if annotations is not None:
            print('This pattern has been identified to match:', pat[0])
            print('The actual pattern in instance is:', pat_full[0])

    return fig